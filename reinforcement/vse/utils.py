from sklearn.utils import shuffle

import pickle
import requests


import plotly.graph_objs as go
import plotly

from plotly.graph_objs import Scatter, Layout
from gensim.models.keyedvectors import KeyedVectors

from logger import Logger
import numpy as np

from datetime import datetime
from config import opt, data, w2v

class ExternalLogger(object):
    def __init__(self, external_logger_name):
        """Create a summary writer logging to log_dir."""
        self.external_logger_name = external_logger_name
        


    def scalar_summary(self, tag, value, step):
        """Log a list of images."""

        logdir = self.external_logger_name
        content = {
            'tag': tag,
            'value': value,
            'step': step,
        }
        url = 'http://masteroppgave.duckdns.org:5000/post_log/{}'.format(logdir)
        res = requests.post(url, json=content)

        # Create and write Summary
        # summary = tf.Summary(value=img_summaries)
        # self.writer.add_summary(summary, step)

def external_logging(external_logger_name):
    lg = ExternalLogger(external_logger_name)
    return lg

def init_logger():
    basename = "./logs/reinforcement"
    lg = Logger('{}-{}'.format(
        basename,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ))

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
        xValues = range(0,len(value))
        data.append(go.Scatter(
            name=classes[key],
            x=list(range(0,len(value))),
            y=value,
            fill='tozeroy'
        ))
    plotly.offline.plot(data, filename=name)

"""
load word2vec pre trained vectors
"""
def load_word2vec():
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
