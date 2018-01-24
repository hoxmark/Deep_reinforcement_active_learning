import heapq
import random
import time

import numpy as np
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

import utils
import train
from scipy import spatial

from config import params, data, w2v


def select_all(model, lg, i):
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

        ret_feature.extend(batch_x)
        ret_target.extend(batch_y)

    return ret_feature, ret_target


def select_egl(model, lg, iteration):
    model.eval()
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
            # Output is not a probability distribution - make it using softmax
            output = nn.functional.softmax(output)

            for index in range(len(data["classes"])):
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
        params["SELECTION_SIZE"], enumerate(sample_scores), key=lambda x: x[1])]

    best_n_scores = [n[1] for n in heapq.nlargest(
        params["SELECTION_SIZE"], enumerate(sample_scores), key=lambda x: x[1])]

    batch_feature = []
    batch_target = []

    for index in sorted(best_n_indexes, reverse=True):
        batch_feature.append([data["word_to_idx"][w] for w in data["train_x"][index]] +
                             [params["VOCAB_SIZE"] + 1 for i in range(params["MAX_SENT_LEN"] - len(data["train_x"][index]))])
        batch_target.append(data["classes"].index(data["train_y"][index]))
        del data["train_x"][index]
        del data["train_y"][index]

    avg_all_score = sum(sample_scores) / len(sample_scores)
    avg_best_score = sum(best_n_scores) / len(best_n_scores)

    if params["LOG"]:
        lg.scalar_summary("avg-score", avg_all_score, iteration)
        lg.scalar_summary("avg-best-score", avg_best_score, iteration)

    return batch_feature, batch_target


def select_entropy(model, lg, iteration):
    model.eval()
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
        # Output is not a probability distribution - make it using softmax
        output = nn.functional.softmax(output)
        output = torch.mul(output, torch.log(output))
        output = torch.sum(output, dim=1)
        output = output * -1

        for s_index, score in enumerate(output):
            sample_scores.append(score.data[0])

        completed += 1
        print("Selection process: {0:.0f}% completed ".format(
            100 * (completed / (len(data["train_x"]) // params["BATCH_SIZE"] + 1))), end="\r")

    sorted_scores_indices = np.flip(np.argsort(sample_scores), 0).tolist()
    batch_indices = []
    batch_feature = []
    batch_target = []
    total_deleted = 0

    for i in range(0, len(sorted_scores_indices), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(sorted_scores_indices) - i)
        next_indices = sorted_scores_indices[i : i + batch_range]

        next_features = [[data["word_to_idx"][w] for w in data["train_x"][index]] +
                        [params["VOCAB_SIZE"] + 1 for i in range(params["MAX_SENT_LEN"] - len(data["train_x"][index]))] for index in next_indices]

        next_targets = [data["classes"].index(data["train_y"][index]) for index in next_indices]


        batch_feature.extend(next_features)
        batch_target.extend(next_targets)
        batch_indices.extend(next_indices)

        print("len before clean {}".format(len(batch_feature)))
        batch_feature, batch_target, batch_indices, n_deleted = clean(batch_feature, batch_target, batch_indices)
        print("len after clean {}".format(len(batch_feature)))
        total_deleted += n_deleted

        if len(batch_feature) >= params["BATCH_SIZE"]:
            break
    # We only want to add batch_size elements each time
    batch_feature = batch_feature[0 : params["BATCH_SIZE"]]
    batch_target = batch_target[0 : params["BATCH_SIZE"]]
    batch_indices = batch_indices[0 : params["BATCH_SIZE"]]

    for index in sorted(batch_indices, reverse=True):
        del data["train_x"][index]
        del data["train_y"][index]

    best_n_scores = [sample_scores[i] for i in batch_indices]
    avg_all_score = sum(sample_scores) / len(sample_scores)
    avg_best_score = sum(best_n_scores) / len(best_n_scores)

    if params["LOG"]:
        lg.scalar_summary("avg-score", avg_all_score, iteration)
        lg.scalar_summary("avg-best-score", avg_best_score, iteration)
        lg.scalar_summary("n-deleted", n_deleted, iteration)

    return batch_feature, batch_target

def clean(features, targets, indices):
    to_delete = []
    for j in range(len(features)):
        for k in range(j + 1, len(features)):
            first = features[j]
            second = features[k]

            first_w2v = utils.average_feature_vector(first, w2v["w2v_kv"])
            second_w2v = utils.average_feature_vector(second, w2v["w2v_kv"])

            distance = spatial.distance.cosine(first_w2v, second_w2v)
            # print("{} < {}".format(distance, params["SIMILARITY_THRESHOLD"]))
            if distance < params["SIMILARITY_THRESHOLD"]:
                to_delete.append(k)
    to_delete = list(set(to_delete))

    print("Deleting {} entries. Feature len is {}".format(len(to_delete), len(features)))
    print(to_delete)
    for delete in sorted(to_delete, reverse=True):
        del features[delete]
        del targets[delete]
        del indices[delete]

    return features, targets, indices, len(to_delete)


def select_random(model, lg, iteration):
    all_sentences = []
    all_targets = []

    for i in range(params["SELECTION_SIZE"]):
        index = random.randint(0, len(data["train_x"]) - 1)
        sentence = [data["word_to_idx"][w]
                    for w in data["train_x"][index]]
        padding = [params["VOCAB_SIZE"] +
                   1 for i in range(params["MAX_SENT_LEN"] - len(sentence))]
        sentence.extend(padding)

        target = data["classes"].index(data["train_y"][index])
        all_sentences.append(sentence)
        all_targets.append(target)
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
