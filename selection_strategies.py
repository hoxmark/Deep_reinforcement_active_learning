import heapq
import random
import time

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

import train


def select_all(model, data, params):
    ret_feature = []
    ret_target = []

    for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

        batch_x = [[data["word_to_idx"][w] for w in sent] +
                   [params["VOCAB_SIZE"] + 1] *
                   (params["MAX_SENT_LEN"] - len(sent))
                   for sent in data["train_x"][i:i + batch_range]]
        batch_y = [data["classes"].index(c)
                   for c in data["train_y"][i:i + batch_range]]

        feature = Variable(torch.LongTensor(batch_x))
        target = Variable(torch.LongTensor(batch_y))
        if params["CUDA"]:
            feature, target = feature.cuda(params["DEVICE"]), target.cuda(params["DEVICE"])
        ret_feature.append(feature)
        ret_target.append(target)

    return ret_feature, ret_target


def select_egl(model, data, params):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    completed = 0
    sample_scores = []
    step = 0

    for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

        batch_x = [[data["word_to_idx"][w] for w in sent] +
                   [params["VOCAB_SIZE"] + 1] *
                   (params["MAX_SENT_LEN"] - len(sent))
                   for sent in data["train_x"][i:i + batch_range]]
        batch_y = [data["classes"].index(c)
                   for c in data["train_y"][i:i + batch_range]]

        feature = Variable(torch.LongTensor(batch_x))
        target = Variable(torch.LongTensor(batch_y))
        if params["CUDA"]:
            feature, target = feature.cuda(
                params["DEVICE"]), target.cuda(params["DEVICE"])

        for s_index in range(batch_range):
            score = 0
            output = model(feature[s_index])
            # TODO: params["NUM_LABELS"]
            for index in range(2):
                optimizer.zero_grad()
                f_target = torch.autograd.Variable(torch.LongTensor([index]))
                if params["CUDA"]:
                    f_target = f_target.cuda(params["DEVICE"])

                loss = criterion(output, f_target)
                loss.backward(retain_graph=True)

                best_grad = -999
                for word in feature[s_index]:
                    grad = model.embed.weight.grad[word.data[0]]
                    grad_length = torch.norm(grad).data[0]
                    best_grad = max(best_grad, grad_length)

                score += output[0][index].data[0] * best_grad
            sample_scores.append(score)

        completed += 1

        print("Selection process: {0:.0f}% completed ".format(
            100 * (completed / (len(data["train_x"]) // params["BATCH_SIZE"] + 1))), end="\r")

    best_n_indexes = [n[0] for n in heapq.nlargest(
        params["BATCH_SIZE"], enumerate(sample_scores), key=lambda x: x[1])]

    batch_feature = []
    batch_target = []

    for index in sorted(best_n_indexes, reverse=True):
        batch_feature.append([data["word_to_idx"][w] for w in data["train_x"][index]] +
                             [params["VOCAB_SIZE"] + 1 for i in range(params["MAX_SENT_LEN"] - len(data["train_x"][index]))])
        batch_target.append(data["train_y"][index])
        del data["train_x"][index]
        del data["train_y"][index]

    return batch_feature, batch_target


def select_entropy(model, data, params):
    sample_scores = []
    completed = 0

    for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)
        batch_x = [[data["word_to_idx"][w] for w in sent] +
                   [params["VOCAB_SIZE"] + 1] *
                   (params["MAX_SENT_LEN"] - len(sent))
                   for sent in data["train_x"][i:i + batch_range]]
        batch_y = [data["classes"].index(c)
                   for c in data["train_y"][i:i + batch_range]]

        feature = Variable(torch.LongTensor(batch_x))
        target = Variable(torch.LongTensor(batch_y))

        if params["CUDA"]:
            feature, target = feature.cuda(
                params["DEVICE"]), target.cuda(params["DEVICE"])

        output = model(feature)

        output = torch.mul(output, torch.log(output))
        output = torch.sum(output, dim=1)
        output = output * -1

        for s_index, score in enumerate(output):
            sample_scores.append(score.data[0])

        completed += 1
        print("Selection process: {0:.0f}% completed ".format(
            100 * (completed / (len(data["train_x"]) // params["BATCH_SIZE"] + 1))), end="\r")

    best_n_indexes = [n[0] for n in heapq.nlargest(
        params["BATCH_SIZE"], enumerate(sample_scores), key=lambda x: x[1])]

    batch_feature = []
    batch_target = []

    for index in sorted(best_n_indexes, reverse=True):
        batch_feature.append([data["word_to_idx"][w] for w in data["train_x"][index]] +
                             [params["VOCAB_SIZE"] + 1 for i in range(params["MAX_SENT_LEN"] - len(data["train_x"][index]))])
        batch_target.append(data["train_y"][index])
        del data["train_x"][index]
        del data["train_y"][index]

    return batch_feature, batch_target


def select_random(model, data, params):
    all_sentences = []
    all_targets = []

    for i in range(params["BATCH_SIZE"]):
        index = random.randint(0, len(data["train_x"]) - 1)
        sentence = [data["word_to_idx"][w]
                    for w in data["train_x"][index]]
        padding = [params["VOCAB_SIZE"] +
                   1 for i in range(params["MAX_SENT_LEN"] - len(sentence))]
        sentence.extend(padding)
        all_sentences.append(sentence)
        all_targets.append(data["train_y"][index])
        del data["train_x"][index]
        del data["train_y"][index]

    return all_sentences, all_targets


def batchify(features, params):
    features = sorted(features, key=lambda x: len(x))

    max_len = 0
    batch_matrix = []

    for feature in features:
        cur_len = 0

        for v_index, value in reversed(list(enumerate(feature))):
            if value.data[0] != 21426:
                cur_len = v_index
                break
        max_len = max(cur_len, max_len)

    for feature in features:
        if params["CUDA"]:
            feature = feature.cuda(params["DEVICE"])

        if(len(feature) < max_len):
            padding = [21426 for x in range(max_len - len(feature))]
            feature.extend(padding)
            padding = torch.LongTensor(padding)

            if params["CUDA"]:
                padding = padding.cuda(params["DEVICE"])
            feature = torch.cat([feature, padding])
        else:
            feature = feature[0:max_len]
        batch_matrix.append(feature)

    batch_tensor = torch.stack(batch_matrix, dim=0)
    return batch_tensor
