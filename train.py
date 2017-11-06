from torch.autograd import Variable
from sklearn.utils import shuffle

import torch
import torch.optim as optim
import torch.nn as nn

from models.cnn import CNN
from models.rnn import RNN
from selection_strategies import select_random, select_entropy, select_egl, select_all


def to_np(x):
    return x.data.cpu().numpy()


def active_train(data, params, lg):
    average_accs = {}
    average_losses = {}
    lg.scalar_summary("test-acc", 0.5, 0)

    if params["MODEL"] == "cnn":
        model = CNN(data, params)
    elif params["MODEL"] == "rnn":
        model = RNN(params, data)
    else:
        model = CNN(data, params)

    if params["CUDA"]:
        model.cuda()

    for j in range(params["N_AVERAGE"]):
        print("-" * 20, "Round {}".format(j), "-" * 20)
        model.init_model()
        train_array = []
        selected_indices = []

        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        n_rounds = 10
        for i in range(n_rounds):

            if params["SCORE_FN"] == "all":
                t1, t2, ret_array = select_all(model, data, selected_indices, params)
                train_array = list(zip(t1, t2))
            # Add a random batch first
            elif i == 0:
                t1, t2, ret_array = select_random(model, data, selected_indices, params)
                train_array.append((t1, t2))
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
            model.init_model()
            train(model, params, train_array, data, lg)
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
            if (params["N_AVERAGE"] == 1):
                lg.scalar_summary("test-acc", accuracy, i + 1)
                lg.scalar_summary("test-loss", loss, i + 1)

    if (params["N_AVERAGE"] != 1):
        for i in range(n_rounds):
            lg.scalar_summary("test-acc", sum(average_accs[i]) / len(average_accs[i]), i + 1)
            lg.scalar_summary("test-loss", sum(average_losses[i]) / len(average_losses[i]), i + 1)

    best_model = {}
    return best_model


def train(model, params, train_array, data, lg):
    print("Length of train set: {}".format(len(train_array)))
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"], weight_decay=params["WEIGHT_DECAY"])
    criterion = nn.CrossEntropyLoss()

    model.train()

    for e in range(params["EPOCH"]):
        avg_loss = 0
        shuffle(train_array)
        corrects = 0
        for feature, target in train_array:
            optimizer.zero_grad()
            pred = model(feature)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.data[0]
            new_corr = (torch.max(pred, 1)[1].view(target.size()).data == target.data).sum()
            corrects += new_corr

        print("Training process: {0:.0f}% completed ".format(100 * (e / params["EPOCH"])), end="\r")


        if params["SCORE_FN"] == "all":
            evaluate(data, model, params, lg, e, mode="dev")
        elif ((e + 1) % 20) == 0:
            # print("Average training loss: {}".format(avg_loss / len(train_array)))
            avg_loss = avg_loss / len(train_array)
            size = len(train_array) * params["BATCH_SIZE"]
            accuracy = 100.0 * corrects / size
            print('{}: Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format("train", avg_loss, accuracy, corrects, size))
            evaluate(data, model, params, lg, e, mode="dev")


def evaluate(data, model, params, lg, step, mode="test"):
    model.eval()

    if params["CUDA"]:
        model.cuda()

    corrects, avg_loss = 0, 0
    for i in range(0, len(data["{}_x".format(mode)]), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(data["{}_x".format(mode)]) - i)

        feature = [[data["word_to_idx"][w] for w in sent] +
                   [params["VOCAB_SIZE"] + 1] *
                   (params["MAX_SENT_LEN"] - len(sent))
                   for sent in data["{}_x".format(mode)][i:i + batch_range]]
        target = [data["classes"].index(c)
                  for c in data["{}_y".format(mode)][i:i + batch_range]]

        feature = Variable(torch.LongTensor(feature))
        target = Variable(torch.LongTensor(target))
        if params["CUDA"]:
            feature = feature.cuda()
            target = target.cuda()

        logit = model(feature)
        loss = torch.nn.functional.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()


    size = len(data["{}_x".format(mode)])
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size

    for tag, value in model.named_parameters():
        if value.requires_grad:
            tag = tag.replace('.', '/')
            lg.histo_summary(tag, to_np(value), step + 1)
            lg.histo_summary(tag + '/grad', to_np(value.grad), step + 1)
    print('{}: Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})\n'.format(mode, avg_loss, accuracy, corrects, size))

    return accuracy, avg_loss
