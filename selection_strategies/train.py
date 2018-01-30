import copy
import logger
import datetime
import utils

from reprint import output
from torch.autograd import Variable
from sklearn.utils import shuffle

import torch
import torch.optim as optim
import torch.nn as nn

from models.cnn import CNN
from models.rnn import RNN
from selection_strategies import select_random, select_entropy, select_egl, select_all

from config import params, data


def to_np(x):
    return x.data.cpu().numpy()


def active_train():
    init_learning_rate = params["LEARNING_RATE"]
    init_selection_size = params["SELECTION_SIZE"]

    init_data = {}
    init_data["train_y"] = copy.deepcopy(data["train_y"])
    init_data["train_x"] = copy.deepcopy(data["train_x"])

    average_accs = {}
    average_losses = {}

    if params["MODEL"] == "cnn":
        model = CNN()
    elif params["MODEL"] == "rnn":
        model = RNN(params, data)
    else:
        model = CNN(data, params)

    if params["CUDA"]:
        model.cuda()

    for j in range(params["N_AVERAGE"]):
        params["LEARNING_RATE"] = init_learning_rate
        params["SELECTION_SIZE"] = init_selection_size

        data["train_x"]  = copy.deepcopy(init_data["train_x"])
        data["train_y"]  = copy.deepcopy(init_data["train_y"])

        lg = None
        if params["LOG"]:
            lg = init_logger(j)
            start_accuracy = 100 / params["CLASS_SIZE"]
            lg.scalar_summary("test-acc", start_accuracy, 0)
            lg.scalar_summary("test-acc-avg", start_accuracy, 0)


        print("-" * 20, "Round {}".format(j + 1), "-" * 20)
        model.init_model()
        train_features = []
        train_targets = []
        distribution = {}

        for key in range(len(data["classes"])):
            distribution[key] = []

        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        if 500 % params["SELECTION_SIZE"] == 0:
            n_rounds = int(500 / params["SELECTION_SIZE"])
            last_selection_size = params["SELECTION_SIZE"]
        else:
            n_rounds = int(500 / params["SELECTION_SIZE"]) + 1
            last_selection_size = 500 % params["SELECTION_SIZE"]

        for i in range(n_rounds):
            if (n_rounds - 1 == i):
                params["SELECTION_SIZE"] = last_selection_size

            if params["SCORE_FN"] == "all":
                t1, t2 = select_all(model, lg, i)
            elif params["SCORE_FN"] == "entropy":
                t1, t2 = select_entropy(model, lg, i)
            elif params["SCORE_FN"] == "egl":
                t1, t2 = select_egl(model, lg, i)
            elif params["SCORE_FN"] == "random":
                t1, t2 = select_random(model, lg, i)

            train_features.extend(t1)
            train_targets.extend(t2)

            print("\n")
            model.init_model()
            model = train(model, train_features, train_targets)
            accuracy, loss, corrects, size = evaluate(model, i, mode="test")
            print("{:10s} loss: {:10.6f} acc: {:10.4f}%({}/{}) \n".format("test", loss, accuracy, corrects, size))
            if i not in average_accs:
                average_accs[i] = [accuracy]
            else:
                average_accs[i].append(accuracy)

            if i not in average_losses:
                average_losses[i] = [loss]

            else:
                average_losses[i].append(loss)

            if params["LOG"]:
                lg.scalar_summary("test-acc", accuracy, len(train_features))
                lg.scalar_summary("test-acc-avg", sum(average_accs[i]) / len(average_accs[i]), len(train_features))

                lg.scalar_summary("test-loss", loss, len(train_features))
                lg.scalar_summary("test-loss-avg", sum(average_losses[i]) / len(average_losses[i]), len(train_features))

                for each in range(len(data["classes"])):
                    val = train_targets.count(each)/len(train_targets)
                    distribution[each].append(val)

                #count number of positive and negativ added to labeledpool.
                nameOfFile = '{}/distribution{}.html'.format(lg.log_dir, j)
                utils.logAreaGraph(distribution, data["classes"], nameOfFile)
                log_model(model, lg)

    best_model = {}
    return best_model


def train(model, train_features, train_targets):
    print("Labeled pool size: {}".format(len(train_features)))

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if params["MODEL"] == "rnn":
        optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"], weight_decay=params["WEIGHT_DECAY"])
    else:
        optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])

    # Softmax is included in CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    model.train()

    best_model = None
    best_acc = 0
    best_epoch = 0

    with output(initial_len=2, interval=0) as output_lines:
        for e in range(params["EPOCH"]):
            shuffle(train_features, train_targets)
            size = len(train_features)
            avg_loss = 0
            corrects = 0

            if params["MINIBATCH"]:
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
                    corrects += (torch.max(pred, 1)[1].view(target.size()).data == target.data).sum()
                avg_loss = avg_loss * params["BATCH_SIZE"] / size
            else:
                feature = Variable(torch.LongTensor(train_features))
                target = Variable(torch.LongTensor(train_targets))

                if params["CUDA"]:
                    feature, target = feature.cuda(params["DEVICE"]), target.cuda(params["DEVICE"])

                optimizer.zero_grad()
                pred = model(feature)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                avg_loss += loss.data[0]
                corrects += (torch.max(pred, 1)[1].view(target.size()).data == target.data).sum()

            if params["SCORE_FN"] == "all":
                accuracy = 100.0 * corrects / size
                dev_accuracy, dev_loss, dev_corrects, dev_size = evaluate(data, model, params, e, mode="dev")

                s1 = "{:10s} loss: {:10.6f} acc: {:10.4f}%({}/{})".format("train", avg_loss, accuracy, corrects, size)
                s2 = "{:10s} loss: {:10.6f} acc: {:10.4f}%({}/{})".format("dev", dev_loss, dev_accuracy, dev_corrects, dev_size)
                output_lines[0] = s1
                output_lines[1] = s2

                if dev_accuracy < best_acc:
                    break

                best_acc = max(dev_accuracy, best_acc)
                best_model = model

            elif ((e + 1) % 10) == 0:
                accuracy = 100.0 * corrects / size
                dev_accuracy, dev_loss, dev_corrects, dev_size = evaluate(model, e, mode="dev")

                s1 = "{:10s} loss: {:10.6f} acc: {:10.4f}%({}/{})".format("train", avg_loss, accuracy, corrects, size)
                s2 = "{:10s} loss: {:10.6f} acc: {:10.4f}%({}/{})".format("dev", dev_loss, dev_accuracy, dev_corrects, dev_size)
                output_lines[0] = s1
                output_lines[1] = s2

                # TODO check if this should also apply for cnn
                if dev_accuracy > best_acc:
                    best_acc = dev_accuracy
                    best_model = copy.deepcopy(model)
                    best_epoch = e

    # return best_model if best_model != None else model
    return best_model


def init_logger(average):
    basename = "./logs" if params["EMBEDDING"] == "static" else "./logs_random"
    if params["MODEL"] == "cnn":
        lg = logger.Logger('{}/cnn/{},minibatch={},selection_size={},date={},FILTERS={},FILTER_NUM={},MODEL={},DROPOUT_EMBED={}, DROPOUT_MODEL={},SCORE_FN={},AVERAGE={},SIMILARITY={}'.format(
            basename,
            str(params["DATASET"]),
            str(params["MINIBATCH"]),
            str(params["SELECTION_SIZE"]),
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            str(params["FILTERS"]),
            str(params["FILTER_NUM"]),
            str(params["MODEL"]),
            str(params["DROPOUT_EMBED"]),
            str(params["DROPOUT_MODEL"]),
            str(params["SCORE_FN"]),
            str(average + 1),
            str(params["SIMILARITY_THRESHOLD"])
        ))

    if (params["MODEL"] == "rnn"):
        lg = logger.Logger('{}/rnn/{},minibatch={},selection_size={},date={},MODEL={},DROPOUT_EMBED={}, DROPOUT_MODEL={},SCORE_FN={},HLAYERS={},HNODES={},AVERAGE={},LEARNING_RATE={},WEIGHT_DECAY={}'.format(
            basename,
            str(params["DATASET"]),
            str(params["MINIBATCH"]),
            str(params["SELECTION_SIZE"]),
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            str(params["MODEL"]),
            str(params["DROPOUT_EMBED"]),
            str(params["DROPOUT_MODEL"]),
            str(params["SCORE_FN"]),
            str(params["HIDDEN_LAYERS"]),
            str(params["HIDDEN_SIZE"]),
            str(average + 1),
            str(params["LEARNING_RATE"]),
            str(params["WEIGHT_DECAY"])
        ))
    return lg


def log_model(model, lg):
    for tag, value in model.named_parameters():
        if value.requires_grad and hasattr(value.grad, "data"):
            tag = tag.replace('.', '/')
            lg.histo_summary(tag, to_np(value), step + 1)
            lg.histo_summary(tag + '/grad', to_np(value.grad), step + 1)


def evaluate(model, step, mode="test"):
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

    return accuracy, avg_loss, corrects, size
