import copy

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
        train_features = []
        train_targets = []

        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        n_rounds = 10
        for i in range(n_rounds):

            print("Unlabeled pool size: {}".format(len(data["train_x"])))
            print("Learning rate: {}".format(params["LEARNING_RATE"]))

            if params["SCORE_FN"] == "all":
                t1, t2 = select_all(model, data, params)
                train_features = list(zip(t1, t2))
            # Add a random batch first
            elif i == 0:
                t1, t2 = select_random(model, data, params)
                train_features.extend(t1)
                train_targets.extend(t2)
            else:
                if params["SCORE_FN"] == "entropy":
                    t1, t2 = select_entropy(model, data, params)
                elif params["SCORE_FN"] == "egl":
                    t1, t2 = select_egl(model, data, params)
                elif params["SCORE_FN"] == "random":
                    t1, t2 = select_random(model, data, params)

                train_features.extend(t1)
                train_targets.extend(t2)

            print("\n")
            model.init_model()
            model = train(model, params, train_features, train_targets, data, lg)
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


def train(model, params, train_features, train_targets, data, lg):
    print("Labeled pool size: {}".format(len(train_features)))

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"], weight_decay=params["WEIGHT_DECAY"])
    criterion = nn.CrossEntropyLoss()
    model.train()

    best_model = None
    best_acc = 0
    best_epoch = 0

    for e in range(params["EPOCH"]):
        shuffle(train_features, train_targets)
        avg_loss = 0
        corrects = 0

        for i in range(0, len(train_features), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(train_features) - i)
            batch_x = train_features[i:i + batch_range]
            batch_y = train_targets[i:i + batch_range]

            feature = Variable(torch.LongTensor(batch_x))
            target = Variable(torch.LongTensor(batch_y))
            if params["CUDA"]:
                feature, target = feature.cuda(params["DEVICE"]), target.cuda(params["DEVICE"])

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
        elif ((e + 1) % 10) == 0:
            avg_loss = avg_loss * params["BATCH_SIZE"] / len(train_features)
            size = len(train_features)
            accuracy = 100.0 * corrects / size
            print('{}: Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format("train", avg_loss, accuracy, corrects, size))
            eval_acc, eval_loss = evaluate(data, model, params, lg, e, mode="dev")

            if eval_acc > best_acc:
                print("New best model at epoch {}".format(e))
                best_acc = eval_acc
                best_model = copy.deepcopy(model)
                best_epoch = e


    # WIMSEN ADAPTIVE LEARNING RATE
    if best_epoch < 60:
        params["LEARNING_RATE"] = params["LEARNING_RATE"] * 0.65

    return best_model


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
        if value.requires_grad and hasattr(value.grad, "data"):
            tag = tag.replace('.', '/')
            lg.histo_summary(tag, to_np(value), step + 1)
            lg.histo_summary(tag + '/grad', to_np(value.grad), step + 1)
    print('{}: Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})\n'.format(mode, avg_loss, accuracy, corrects, size))

    return accuracy, avg_loss
