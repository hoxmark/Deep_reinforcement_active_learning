from torch.autograd import Variable
from sklearn.utils import shuffle

import torch
import torch.optim as optim
import torch.nn as nn

from model import CNN
from selection_strategies import select_random, select_entropy, select_egl


def to_np(x):
    return x.data.cpu().numpy()


def active_train(data, params, lg):
    average_accs = {}
    average_losses = {}
    lg.scalar_summary("test-acc", 0, 0)

    for j in range(params["N_AVERAGE"]):
        model = CNN(data, params)
        if params["CUDA"]:
            model.cuda(params["DEVICE"])

        train_array = []
        selected_indices = []

        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        n_rounds = 25
        for i in range(n_rounds):
            # Add a random batch first
            if i == 0:
                t1, t2, ret_array = select_random(model, data, selected_indices, params)
            else:
                if params["SCORE_FN"] == "entropy":
                    t1, t2, ret_array = select_entropy(model, data, selected_indices, params)
                elif params["SCORE_FN"] == "egl":
                    t1, t2, ret_array = select_egl(model, data, selected_indices, params)
                elif params["SCORE_FN"] == "random":
                    t1, t2, ret_array = select_random(model, data, selected_indices, params)

            train_array.append((t1, t2))
            selected_indices.extend(ret_array)

            print("\n")

            train(model, params, train_array)
            accuracy, loss = evaluate(data, model, params, lg, i, mode="dev")
            if i not in average_accs:
                average_accs[i] = [accuracy]
            else:
                average_accs[i].append(accuracy)

            if i not in average_losses:
                average_losses[i] = [loss]

            else:
                average_losses[i].append(loss)

            print("New  accuracy: {}".format(sum(average_accs[i]) / len(average_accs[i])))

    for i in range(n_rounds):
        lg.scalar_summary("test-acc", sum(average_accs[i]) / len(average_accs[i]), i + 1)
        lg.scalar_summary("test-loss", sum(average_losses[i]) / len(average_losses[i]), i + 1)

    best_model = {}
    return best_model


def train(model, params, train_array):
    print("Length of train set: {}".format(len(train_array)))
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    for e in range(params["EPOCH"]):
        for feature, target in train_array:
            optimizer.zero_grad()
            model.train()
            pred = model(feature)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            # constrain l2-norms of the weight vectors
            if model.fc.weight.norm().data[0] > params["NORM_LIMIT"]:
                model.fc.weight.data = model.fc.weight.data * \
                    params["NORM_LIMIT"] / model.fc.weight.data.norm()


def evaluate(data, model, params, lg, step, mode="test"):
    model.eval()
    if params["CUDA"]:
        model.cuda(params["DEVICE"])

    corrects, avg_loss = 0, 0
    for i in range(0, len(data["dev_x"]), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(data["dev_x"]) - i)

        feature = [[data["word_to_idx"][w] for w in sent] +
                   [params["VOCAB_SIZE"] + 1] *
                   (params["MAX_SENT_LEN"] - len(sent))
                   for sent in data["dev_x"][i:i + batch_range]]
        target = [data["classes"].index(c)
                  for c in data["dev_y"][i:i + batch_range]]

        feature = Variable(torch.LongTensor(feature))
        target = Variable(torch.LongTensor(target))
        if params["CUDA"]:
            feature = feature.cuda(params["DEVICE"])
            target = target.cuda(params["DEVICE"])

        logit = model(feature)
        loss = torch.nn.functional.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data["dev_x"])
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size

    for tag, value in model.named_parameters():
        if value.requires_grad:
            tag = tag.replace('.', '/')
            lg.histo_summary(tag, to_np(value), step + 1)
            lg.histo_summary(tag + '/grad', to_np(value.grad), step + 1)
    print(
        'Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})\n'.format(avg_loss, accuracy, corrects, size))

    return accuracy, avg_loss
