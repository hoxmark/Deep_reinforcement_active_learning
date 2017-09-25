

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


def key_func2(t):
    b_index, index, target, score = t
    return score


def key_func(sample):
    feature, target, score = sample[1]
    return score


def select_n_best_samples(model, optimizer, train_iter, n_samples, already_selected, args, text_field):
    sample_scores = []
    completed = 0
    slide_n = 500
    print_every = 500
    batch_scores = [0 for i in range(slide_n)]
    num_1 = 0
    num_0 = 0
    # total_tensor = torch.FloatTensor(len(train_iter) * args.batch_size, 44).zero_()

    all_tensor = []

    max_len = 0

    for b_index, batch in enumerate(train_iter):
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align

        if args.cuda:
            model.cuda()
            feature, target = feature.cuda(), target.cuda()

        output = model(feature)
        output = torch.mul(output, torch.log(output))
        output = torch.sum(output, dim=1)
        output = output * -1

        for index, score in enumerate(output):
            if((b_index, index) in already_selected):
                # print("SKIPPING")
                pass
            # all_tensor.append([x for x in feature.data[index]])
            all_tensor.append(feature.data[index])
            # print(len(feature.data[index]))
            max_len = max(len(feature.data[index]), max_len)
            sample_scores.append(
                (b_index, index, target.data[index], score.data[0]))
            batch_scores.pop(0)
            batch_scores.append(score.data[0])
        completed += 1

        if b_index % print_every == 0:
            print("Sliding average: {}".format(sum(batch_scores) / slide_n))
        print("Selection process: {0:.2f}% completed".format(
            100 * (completed / len(train_iter))), end="\r")

    best_n_indexes = heapq.nlargest(n_samples, sample_scores, key_func2)
    # best_n_indexes = random.sample(sample_scores, n_samples)

    ret_batch = []
    ret_batch_target = []


    for b_index, index, target, score in best_n_indexes:
        if target == 1:
            num_1 += 1
        if target == 0:
            num_0 += 1

        i = args.batch_size * b_index + index
        # ones = torch.ones(max_len - len(all_tensor[i]))
        ones = [1 for x in range(max_len - len(all_tensor[i]))]
        ones = torch.LongTensor(ones)
        if args.cuda:
            ones = ones.cuda()
        all_tensor[i] = torch.cat([all_tensor[i], ones])

        ret_batch.append(all_tensor[i])
        # print(len(all_tensor[i]))
        ret_batch_target.append(target)

        # torch.nn.functional.pad
    print("0's: {}, 1's: {}".format(num_0, num_1))
    ret_batch_target = torch.LongTensor(ret_batch_target)
    if args.cuda:
        ret_batch_target = ret_batch_target.cuda()

    ret_batch = torch.stack(ret_batch, dim=0)
    # print(ret_batch)

    return torch.autograd.Variable(ret_batch), torch.autograd.Variable(ret_batch_target), [(b_index, index) for b_index, index, target, score in best_n_indexes]


def active_train(train_iter, dev_iter, model, args, text_field):
    torch.set_printoptions(profile="full")

    already_selected = []
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_set = []


    # Add some random data to begin with
    for index, batch in enumerate(train_iter):
        if index == 1:
            break
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        train_set.append((feature, target, 0))
    for i in range(20):
        t1, t2, added_indices = select_n_best_samples(
            model, optimizer, train_iter, args.batch_size, already_selected, args, text_field)
        train_set.append((t1, t2, 0))
        already_selected.extend(added_indices)


        # Reset the model and train again
        model = CNNModel.CNN_Text(args)
        print("Selected {} best samples. Training".format(args.batch_size))
        print("Length of train set: ", len(train_set))
        train(train_set, model, args, dev_iter)
        print("Finished training ")
        eval(dev_iter, model, args)
        # if not os.path.isdir(args.save_dir):
        # os.makedirs(args.save_dir)
        # save_prefix = os.path.join(args.save_dir, 'snapshot')
        # save_path = '{}_steps{}.pt'.format(save_prefix, i)
        # torch.save(model, save_path)


def train(train_iter, model, args, dev_iter):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    if args.cuda:
        model.cuda()
    for i in range(args.epochs):
        for sample in train_iter:
            feature, target, score = sample

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            # If we have samples of size 1, unsqueeze it to [1 x len(feature)]
            # dimension
            # if (len(feature.size()) == 1):
            #     feature = feature.unsqueeze(0)

            optimizer.zero_grad()
            output = model(feature)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        if i % 20 == 0:
            print("Batch {}: ".format(i * len(train_iter)))
            eval(dev_iter, model, args)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))


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
