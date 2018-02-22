from sklearn.utils import shuffle

import pickle
import time


import plotly.graph_objs as go
import plotly
from plotly.graph_objs import Scatter, Layout
from gensim.models.keyedvectors import KeyedVectors

from config import data, params, w2v
import numpy as np

def read_TREC():
    def read(mode):
        z = []
        with open("{}/TREC/TREC_".format(params['DATA_PATH']) + mode + ".txt", "r", encoding="utf-8") as f:
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

    return data


def read_MR():
    # data = {}
    x, y = [], []

    with open("{}/MR/rt-polarity.pos".format(params['DATA_PATH']), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("{}/MR/rt-polarity.neg".format(params['DATA_PATH']), "r", encoding="utf-8") as f:
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

def read_MR7025():
    # data = {}
    x, y = [], []

    with open("{}/MR/rt-polarity.pos".format(params['DATA_PATH']), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("{}/MR/rt-polarity-small.neg".format(params['DATA_PATH']), "r", encoding="utf-8") as f:
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

def read_rotten_imdb():
    # data = {}
    x, y = [], []

    with open("{}/rotten_imdb/rt-polarity.pos".format(params['DATA_PATH']), "r", encoding="ISO-8859-1") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("{}/rotten_imdb/rt-polarity.neg".format(params['DATA_PATH']), "r", encoding="ISO-8859-1") as f:
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
    # data = {}
    x, y = [], []

    with open("{}/UMICH/rt-polarity.pos".format(params['DATA_PATH']), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("{}/UMICH/rt-polarity.neg".format(params['DATA_PATH']), "r", encoding="utf-8") as f:
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


def save_model(model, params):
    path = "saved_models/{}_{}_{}.pkl".format(params['DATASET'], params['MODEL'], params['EPOCH'])
    pickle.dump(model, open(path, "wb"))
    print("A model is saved successfully as {}!".format(path))


def load_model(params):
    path = "saved_models/{}_{}_{}.pkl".format(params['DATASET'], params['MODEL'], params['EPOCH'])

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
    t0 = time.time()
    
    word_vectors = KeyedVectors.load_word2vec_format(
        "{}/GoogleNews-vectors-negative300.bin".format(params['DATA_PATH']), binary=True)
    
    t1 = time.time()

    total = t1-t0

    print("It took {} to load w2v".format(total))

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


def average_feature_vector(sentence, embedding):
    num_features = 300
    feature_vector = np.zeros((num_features, ), dtype="float32")

    for word in sentence:
        # word_vector = embedding(word)
        if word in embedding.vocab:
            word_vector = embedding.word_vec(word)
        else:
            word_vector = np.random.uniform(-0.01, 0.01, num_features).astype("float32")
        feature_vector = np.add(feature_vector, word_vector)

    feature_vector = np.divide(feature_vector, len(sentence))
    return feature_vector
