from model import CNN
import utils
import heapq
import random
import logger
import datetime

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy

from selection_strategies import *


def to_np(x):
    return x.data.cpu().numpy()


def train(data, params, lg):
    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format(
            "GoogleNews-vectors-negative300.bin", binary=True)

        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(
                    np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    model = CNN(**params)
    if params["CUDA"]:
        model.cuda(params["DEVICE"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    train_array = []
    selected_indices = []

    data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

    for i in range(25):
        t1, t2, ret_array = select_entropy(
            model, data, selected_indices, params)
        train_array.append((t1, t2))
        selected_indices.extend(ret_array)
        #
        print("\n")
        if params["RESET"]:
            model = CNN(**params)
            if params["CUDA"]:
                model.cuda(params["DEVICE"])

        print("Length of train set: {}".format(len(train_array)))
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

        test(data, model, params, lg, i, mode="dev")

    best_model = {}
    return best_model


def test(data, model, params, lg, step, mode="test"):
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
        loss = torch.nn.functional.cross_entropy(
            logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data["dev_x"])
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    lg.scalar_summary("test-acc", accuracy, step + 1)
    lg.scalar_summary("test-loss", avg_loss, step + 1)
    for tag, value in model.named_parameters():
        if value.requires_grad:
            tag = tag.replace('.', '/')
            lg.histo_summary(tag, to_np(value), step + 1)
            lg.histo_summary(tag + '/grad', to_np(value.grad), step + 1)
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})\n'.format(avg_loss,
                                                                    accuracy,
                                                                    corrects,
                                                                    size))
