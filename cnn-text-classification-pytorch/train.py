import numpy
import heapq
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import model as CNNModel


def key_func(sample):
    feature, target, score = sample
    return score


def select_n_best_samples(model, optimizer, train_iter, n_samples, args):
    sample_scores = []
    completed = 0
    for batch in train_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        for s_index, sentence in enumerate(feature):
            output = model(sentence.unsqueeze(0))
            score = 0

            for index, k in enumerate(output):
                loss = F.cross_entropy(output, torch.autograd.Variable(torch.LongTensor([index])))
                loss.backward(retain_graph=True)

                best_grad = -999
                for word in sentence:
                    grad = model.embed.weight.grad[word]
                    grad_max = grad.max().data[0]
                    if grad_max > best_grad:
                        best_grad = grad_max
                score = score + k * best_grad

            sample_scores.append((sentence, target[s_index], 0))
            completed = completed + 1
        # if completed > 1200:
            # break
        print("Done with {} of {}".format(completed, args.batch_size * len(train_iter)), end="\r")
    best_n_samples = heapq.nlargest(n_samples, sample_scores, key_func)
    return best_n_samples


def active_train(train_iter, dev_iter, model, args):
    torch.set_printoptions(profile="full")
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_set = []
    for index, batch in enumerate(train_iter):
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        train_set.append((feature, target, 0))
        break

    print("Length of train set: ", len(train_set))
    train(train_set, model, args)
    print("Finished with random sample training")
    eval(dev_iter, model, args)
    for i in range(args.batch_size):
        feature_matrix = []
        target_matrix = []
        n_best_samples = select_n_best_samples(
            model, optimizer, train_iter, args.batch_size, args)
        train_set.extend(n_best_samples)
        # Reset the model and train again
        model = CNNModel.CNN_Text(args)
        print("Selected 25 best samples. Training")
        print("Length of train set: ", len(train_set))
        train(train_set, model, args)
        print("Finished training ")

        eval(dev_iter, model, args)
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        save_prefix = os.path.join(args.save_dir, 'snapshot')
        save_path = '{}_steps{}.pt'.format(save_prefix, i)
        torch.save(model, save_path)


def train(train_iter, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for sample in train_iter:
        feature, target, score = sample

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        # If we have samples of size 1, unsqueeze it to [1 x Y] dimension
        if (len(feature.size()) == 1):
            feature = feature.unsqueeze(0)

        optimizer.zero_grad()
        output = model(feature)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


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
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0] + 1]
