import heapq
import random

from torch.autograd import Variable
import torch


def select_entropy(model, data, selected_indices, params):
    if params["EVAL"]:
        model.eval()

    sample_scores = []
    all_tensors = []
    all_targets = []

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

        all_tensors.extend(feature)
        all_targets.extend(target)

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

    batch_features = []
    batch_target = []

    for index in best_n_indexes:
        batch_features.append(all_tensors[index])
        batch_target.append(all_targets[index].data[0])

    batch_feature = torch.stack(batch_features, dim=0)
    batch_target = torch.autograd.Variable(torch.LongTensor(batch_target))

    if params["CUDA"]:
        batch_feature = batch_feature.cuda(params["DEVICE"])
        batch_target = batch_target.cuda(params["DEVICE"])

    return batch_feature, batch_target, best_n_indexes


def select_random(model, data, selected_indices, params):
    if params["EVAL"]:
        model.eval()

    all_sentences = []
    all_targets = []

    for i in range(params["BATCH_SIZE"]):
        random_idx = random.randint(0, len(data["train_x"]))
        sentence = [data["word_to_idx"][w]
                    for w in data["train_x"][random_idx]]
        padding = [params["VOCAB_SIZE"] +
                   1 for i in range(params["MAX_SENT_LEN"] - len(sentence))]
        sentence.extend(padding)
        all_sentences.append(sentence)
        all_targets.append(data["train_y"][random_idx])

    batch_feature = torch.autograd.Variable(torch.LongTensor(all_sentences))
    batch_target = torch.autograd.Variable(torch.LongTensor(all_targets))

    if params["CUDA"]:
        batch_feature = batch_feature.cuda(params["DEVICE"])
        batch_target = batch_target.cuda(params["DEVICE"])

    return batch_feature, batch_target, []


def batchify(features, params):
    features = sorted(features, key=lambda x: len(x))
    # print(features)

    max_len = 0
    batch_matrix = []

    for feature in features:
        # print(feature)
        cur_len = 0

        for v_index, value in reversed(list(enumerate(feature))):
            # print(value.data[0])
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
