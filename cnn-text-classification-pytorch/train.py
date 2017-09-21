
from __future__ import division
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


def key_func(sample):
    feature, target, score = sample[1]
    return score


def select_n_best_samples(model, optimizer, train_iter, n_samples, already_selected, args):
    sample_scores = []
    completed = 0
    slide_n = 500
    batch_scores = [0 for i in range(slide_n)]

    for b_index, batch in enumerate(train_iter):
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align

        if args.cuda:
            model.cuda()
            feature, target = feature.cuda(), target.cuda()
        # model_copy = model

        # model_copy = CNNModel.CNN_Text(args)
        # model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
        #
        # if(args.cuda):
        #     model_copy.cuda()

        for s_index, sentence in enumerate(feature):

            # model_copy = model

            if(b_index * args.batch_size + s_index in already_selected):
                print("WE ARE PASSING ")
                pass
            optimizer.zero_grad()
            output = model(sentence.unsqueeze(0))
            score = 0
            for index, k in enumerate(output.data[0]):
                # f_target = torch.autograd.Variable(torch.LongTensor([index]))
                #
                # if args.cuda:
                #     f_target = f_target.cuda()
                # loss = F.cross_entropy(output, f_target)
                # loss.backward(retain_graph=True)
                #
                # best_grad = -999
                # for word in sentence:
                #     grad = model_copy.embed.weight.grad[word]
                #     grad_max = grad.max().data[0]
                #     if grad_max > best_grad:
                #         best_grad = grad_max
                # score = score + k * abs(best_grad)
                # print(k)
                score = score + k * math.log(k)

            sample_scores.append((sentence, target[s_index], score * -1))
            batch_scores.pop(0)
            batch_scores.append(score)
            completed = completed + 1
        print("Sliding average: {}".format(sum(batch_scores) / slide_n))

        print("Selection process: {0:.2f}% completed".format(
            100 * (completed / (args.batch_size * len(train_iter)))), end="\r")
    best_n_samples = heapq.nlargest(
        n_samples, enumerate(sample_scores), key_func)
    delete_indices = [s[0] for s in best_n_samples]

    return ([s[1] for s in best_n_samples], delete_indices)


def active_train(train_iter, dev_iter, model, args):
    torch.set_printoptions(profile="full")

    already_selected = []
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_set = []

    # Add some random data to begin with
    for index, batch in enumerate(train_iter):
        if index > 1:
            break
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        train_set.append((feature, target, 0))

    print("Length of train set: ", len(train_set))
    train(train_set, model, args, dev_iter)
    print("Finished with random sample training")
    # eval(dev_iter, model, args)
    for i in range(args.batch_size):
        feature_matrix = []
        target_matrix = []
        n_best_samples, delete_indices = select_n_best_samples(
            model, optimizer, train_iter, args.batch_size, already_selected, args)
        already_selected.extend(delete_indices)

        max_length = 0
        for feature, target, s in n_best_samples:
            sentence = [x.data[0] for x in feature]
            max_length = max(max_length, len(sentence))

            feature_matrix.append(sentence)
            target_matrix.append(target.data[0])

        for sentence in feature_matrix:
            for i in range(max_length - len(sentence)):
                sentence.append(1)

        t1 = torch.autograd.Variable(torch.LongTensor(feature_matrix))
        t2 = torch.autograd.Variable(torch.LongTensor(target_matrix))

        train_set.append((t1, t2, 0))

        # train_set.extend(n_best_samples)

        # Reset the model and train again
        # model = CNNModel.CNN_Text(args)
        print("Selected {} best samples. Training".format(args.batch_size))
        print("Length of train set: ", len(train_set))
        train(train_set, model, args, dev_iter)
        print("Finished training ")

        # eval(dev_iter, model, args)
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        save_prefix = os.path.join(args.save_dir, 'snapshot')
        save_path = '{}_steps{}.pt'.format(save_prefix, i)
        torch.save(model, save_path)


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
            if (len(feature.size()) == 1):
                feature = feature.unsqueeze(0)

            optimizer.zero_grad()
            output = model(feature)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        if i % 50 == 0:
            print("Batch {}: ".format(i * len(train_iter) * args.batch_size))
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
