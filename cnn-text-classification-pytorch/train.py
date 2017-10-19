from __future__ import division
import random
import copy
import math
import numpy
import heapq
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import model as CNNModel
import logger
import datetime

def key_func2(t):
    b_index, index, target, score = t
    return score


def key_func(sample):
    feature, target, score = sample[1]
    return score


def select_n_best_samples(model, train_array, n_samples, args):
    if args.eval:
        model.eval()
    sample_scores = []
    completed = 0
    slide_n = 500
    print_every = 500
    sliding_scores = [0 for i in range(slide_n)]

    all_tensor = []

    max_len = 0

    for b_index, (feature, target) in enumerate(train_array):
        if args.cuda:
            model.cuda(args.device)
            feature, target = feature.cuda(args.device), target.cuda(args.device)
        output = model(feature)
        output = torch.mul(output, torch.log(output))
        output = torch.sum(output, dim=1)
        output = output * -1

        # l = len(target.data)
        # for s_index, score in enumerate([x for x in range(l)]):
        for s_index, score in enumerate(output):
            sample_scores.append(score.data[0])
            # sample_scores.append(0)
        completed += 1

        print("Selection process: {0:.2f}% completed ".format(
            100 * (completed / len(train_array))), end="\r")

    print("\n")
    best_n_indexes = [n[0] for n in heapq.nlargest(n_samples, enumerate(sample_scores), key=lambda x: x[1])]
    # best_n_indexes = [n[0] for n in random.sample(list(enumerate(sample_scores)), args.batch_size)]

    batch_features = []
    batch_target = []

    for index in best_n_indexes:
        batch_index = index // args.batch_size
        s_index = index % args.batch_size

        batch_features.append(train_array[batch_index][0][s_index])
        batch_target.append(train_array[batch_index][1][s_index].data[0])

    batch_feature = batchify(batch_features, args)
    batch_target = torch.autograd.Variable(torch.LongTensor(batch_target))

    return batch_feature, batch_target, train_array


def batchify(features, args):
    # features.sort(key=)
    features = sorted(features, key=lambda x: len(x))
    # print(features)

    max_len = 0
    batch_matrix = []

    for feature in features:
        cur_len = 0

        for v_index, value in reversed(list(enumerate(feature.data))):
            if value != 1:
                cur_len = v_index
                break
        max_len = max(cur_len, max_len)

    for feature in features:
        if args.cuda:
            feature = feature.cuda(args.device)

        if(len(feature) < max_len):
            padding = [1 for x in range(max_len - len(feature))]
            padding = torch.LongTensor(padding)

            if args.cuda:
                padding = padding.cuda(args.device)
            feature = torch.cat([feature, padding])
        else:
            feature = feature[0:max_len]
        batch_matrix.append(feature)

    batch_tensor = torch.stack(batch_matrix, dim=0)
    return batch_tensor
    # return torch.nn.utils.rnn.pack_padded_sequence(features, [len(x) for x in features])


def active_train(train_array, dev_array, model, args, text_field, lg):
    torch.set_printoptions(profile="full")

    model = CNNModel.CNN_Text(args)

    if args.cuda:
        model = model.cuda(args.device)

    already_selected = []
    if args.cuda:
        model.cuda(args.device)

    # torch.optim.

    train_set = []
    # for i in range(21):
    init_batch_feature, init_batch_targets = random.choice(train_array)
    train_set.append((batchify(init_batch_feature, args), init_batch_targets))

    train(train_set, model, args, dev_array)
    eval(dev_array, model, args, len(train_set), lg)

    for i in range(25):
        t1, t2, train_array = select_n_best_samples(
            model, train_array, args.batch_size, args)
        train_set.append((t1, t2))

        # init_batch_feature, init_batch_targets = random.choice(train_array)
        # train_set.append((batchify(init_batch_feature, args), init_batch_targets))

        print("Length of train set: {}".format(len(train_set)))

        if args.reset:
            model = CNNModel.CNN_Text(args)
            if args.cuda:
                model.cuda()
        model.train()
        train(train_set, model, args, dev_array)
        eval(dev_array, model, args, len(train_set), lg)
    # if not os.path.isdir(args.save_dir):
    # os.makedirs(args.save_dir)
    # save_prefix = os.path.join(args.save_dir, 'snapshot')
    # save_path = '{}_steps{}.pt'.format(save_prefix, i)
    # torch.save(model, save_path)


def train(train_array, model, args, dev_array):
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adadelta(model.parameters())
    steps = 0

    if args.cuda:
        model.cuda(args.device)
    for i in range(args.epochs):
        for feature, target in train_array:

            # print(feature)
            # print(target)

            if args.cuda:
                feature, target = feature.cuda(args.device), target.cuda(args.device)


            for index, sentence in enumerate(feature):
                # print(sentence.unsqueeze(0))

                optimizer.zero_grad()
                output = model(sentence.unsqueeze(0))
                loss = F.cross_entropy(output, target[index])
                loss.backward()
                optimizer.step()
            steps += 1
            if steps % 20 == 0:
                eval(dev_array, model, args, -1, None)
    print("\n")

def to_np(x):
    return x.data.cpu().numpy()

def eval(data_iter, model, args, step, lg):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(args.device), target.cuda(args.device)

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    if (step != -1):
        lg.scalar_summary("eval-acc", accuracy, step+1)
        lg.scalar_summary("eval-loss", avg_loss, step+1)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            lg.histo_summary(tag, to_np(value), step+1)
            lg.histo_summary(tag+'/grad', to_np(value.grad), step+1)
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss,
                                                                  accuracy,
                                                                  corrects,
                                                                  size), end="\r")


def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    print(text)
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0] + 1]
