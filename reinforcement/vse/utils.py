from sklearn.utils import shuffle

import pickle
import requests
import time
import json
import copy

import plotly.graph_objs as go
import plotly

from plotly.graph_objs import Scatter, Layout
from gensim.models.keyedvectors import KeyedVectors

from logger import Logger, ExternalLogger, NoLogger
import numpy as np

from datetime import datetime
from config import opt, data, w2v


def timer(func, args):
    """Timer function to time the duration of a spesific function func """
    time1 = time.time()
    ret = func(*args)
    time2 = time.time()
    ms = (time2 - time1) * 1000.0
    print("{}() in {:.2f} ms".format(func.__name__, ms))
    return ret


def no_logger():     
    """function that return an logger-object that will just discard everything sent to it.
    This if for testing purposes, so we don't fill up the logs with test data"""                       
    lg = NoLogger()
    return lg


def external_logging(external_logger_name): 
    """function that return an logger-object to sending tensorboard logs to external server"""
    lg = ExternalLogger(external_logger_name)
    return lg


def init_logger():      
    """function that return an logger-object to saving tensorboard logs locally"""
    basename = "./logs/reinforcement"
    nameoffolder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    lg = Logger('{}-{}'.format(
        basename,
        nameoffolder
    ))

    #need to remove the vocab object from opt because its not JSON serializable
    with open('{}-{}/parameters.json'.format(basename,nameoffolder), 'w') as outfile:
        vocab = opt.vocab 
        opt.vocab = 'removedFromDump' 
        json.dump(opt, outfile, sort_keys=True, indent=4, separators=(',', ': '))
        opt.vocab = vocab
    return lg


def read_TREC():
    def read(mode):
        z = []
        with open("{}/TREC/TREC_".format(opt.data_path) + mode + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                feature = " ".join(line.split()[1:])
                target = line.split()[0].split(":")[0]
                z.append((feature, target))

        # Remove duplicates
        z = list(set(z))
        x = [tup[0].split() for tup in z]
        y = [tup[1] for tup in z]

        x, y = shuffle(x, y)

        if mode == "train":
            dev_idx = len(x) // 10
            data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
            data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")


def read_MR():
    x, y = [], []

    with open("{}/MR/rt-polarity.pos".format(opt.data_path), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("{}/MR/rt-polarity.neg".format(opt.data_path), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]


def read_MR7025():
    x, y = [], []

    with open("{}/MR/rt-polarity.pos".format(opt.data_path), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("{}/MR/rt-polarity-small.neg".format(opt.data_path), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]


def read_rotten_imdb():
    data = {}
    x, y = [], []

    with open("{}/rotten_imdb/rt-polarity.pos".format(opt.data_path), "r", encoding="ISO-8859-1") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("{}/rotten_imdb/rt-polarity.neg".format(opt.data_path), "r", encoding="ISO-8859-1") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def read_UMICH():
    data = {}
    x, y = [], []

    with open("{}/UMICH/rt-polarity.pos".format(opt.data_path), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("{}/UMICH/rt-polarity.neg".format(opt.data_path), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def save_model(model):
    path = "saved_models/{}_{}_{}.pkl".format(opt.dataset, opt.model, opt.epoch)
    pickle.dump(model, open(path, "wb"))
    print("A model is saved successfully as {}!".format(path))


def load_model():
    path = "saved_models/{}_{}_{}.pkl".format(opt.dataset, opt.model, opt.epoch)

    try:
        model = pickle.load(open(path, "rb"))
        print("Model in {} loaded successfully!".format(path))

        return model
    except:
        print("No available model such as {}.".format(path))
        exit()


def logAreaGraph(distribution, classes, name):
    data = []
    for key, value in distribution.items():
        xValues = range(0, len(value))
        data.append(go.Scatter(
            name=classes[key],
            x=list(range(0, len(value))),
            y=value,
            fill='tozeroy'
        ))
    plotly.offline.plot(data, filename=name)


def load_word2vec():
    """Load word2vec pre trained vectors"""
    print("loading word2vec...")
    word_vectors = KeyedVectors.load_word2vec_format(
        "{}/GoogleNews-vectors-negative300.bin".format(opt.data_path), binary=True)

    # data["w2v_kv"] = word_vectors

    wv_matrix = []
    for word in data["vocab"]:
        if word in word_vectors.vocab:
            wv_matrix.append(word_vectors.word_vec(word))
        else:
            wv_matrix.append(
                np.random.uniform(-0.01, 0.01, 300).astype("float32"))

    # one for UNK and one for zero padding
    wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    wv_matrix.append(np.zeros(300).astype("float32"))
    wv_matrix = np.array(wv_matrix)
    w2v["w2v"] = wv_matrix
    w2v["w2v_kv"] = word_vectors
    # return word_vectors, wv_matrix
