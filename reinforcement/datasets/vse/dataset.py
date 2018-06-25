from datasets.vse.vocab import Vocabulary
import os
import numpy as np
import nltk
import torch
import pickle

from config import opt
def load_data():
    vocab = pickle.load(open(os.path.join(opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab = vocab
    opt.vocab_size = len(vocab)


    train_captions = []
    dev_captions = []
    test_captions = []

    with open("{}/{}/dev_caps.txt".format(opt.data_path, opt.data_name)) as f:
        for line in f:
            dev_captions.append(line.strip())

    with open("{}/{}/train_caps.txt".format(opt.data_path, opt.data_name)) as f:
        for line in f:
            train_captions.append(line.strip())

    with open("{}/{}/test_caps.txt".format(opt.data_path, opt.data_name)) as f:
        for line in f:
            test_captions.append(line.strip())

    train_images = np.load("{}/{}/train_ims.npy".format(opt.data_path, opt.data_name))
    dev_images = np.load("{}/{}/dev_ims.npy".format(opt.data_path, opt.data_name))
    test_images = np.load("{}/{}/test_ims.npy".format(opt.data_path, opt.data_name))

    train_tokens = []
    dev_tokens = []
    test_tokens = []

    for caption in train_captions:
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        cap = []
        cap.append(opt.vocab('<start>'))
        cap.extend([opt.vocab(token) for token in tokens])
        cap.append(opt.vocab('<end>'))
        train_tokens.append(cap)

    for caption in dev_captions:
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        cap = []
        cap.append(opt.vocab('<start>'))
        cap.extend([opt.vocab(token) for token in tokens])
        cap.append(opt.vocab('<end>'))
        dev_tokens.append(cap)

    for caption in test_captions:
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        cap = []
        cap.append(opt.vocab('<start>'))
        cap.extend([opt.vocab(token) for token in tokens])
        cap.append(opt.vocab('<end>'))
        test_tokens.append(cap)

    train_cap_lengths = [len(cap) for cap in train_tokens]
    dev_cap_lengths = [len(cap) for cap in dev_tokens]
    test_cap_lengths = [len(cap) for cap in test_tokens]

    def pad(captions, lengths):
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            cap = torch.Tensor(cap)
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return targets.cpu().numpy()

    sort_idx = np.argsort(-1 * np.array(train_cap_lengths))
    train_cap_lengths = np.array(train_cap_lengths)[sort_idx]
    train_images = np.array(train_images)[sort_idx]
    train_tokens = np.array(train_tokens)[sort_idx]

    sort_idx = np.argsort(-1 * np.array(dev_cap_lengths))
    dev_cap_lengths = np.array(dev_cap_lengths)[sort_idx]
    dev_images = np.array(dev_images)[sort_idx]
    dev_tokens = np.array(dev_tokens)[sort_idx]

    sort_idx = np.argsort(-1 * np.array(test_cap_lengths))
    test_cap_lengths = np.array(test_cap_lengths)[sort_idx]
    test_images = np.array(test_images)[sort_idx]
    test_tokens = np.array(test_tokens)[sort_idx]

    train_data = (train_images, pad(train_tokens, train_cap_lengths), train_cap_lengths)
    dev_data = (dev_images, pad(dev_tokens, dev_cap_lengths), dev_cap_lengths)
    test_data = (test_images, pad(test_tokens, test_cap_lengths), test_cap_lengths)

    opt.data_size = opt.topk
    opt.pred_size = opt.topk
    opt.data_len = len(train_images)

    return (train_data, dev_data, test_data)
