from pprint import pprint
import utils
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

from scipy import spatial

num_features = 300


def average_distance_per_category(features, targets, classes):
    print("Loading w2v")
    w2v = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)


    print("Calculating")
    tuples = list(zip(features, targets))
    avg_similarity = {}

    for c in classes:
        features = [tup[0] for tup in tuples if tup[1] == c]
        feature_vector = np.zeros((num_features, ), dtype="float32")

        sentence_vectors = []

        for sentence in features:
            sentence_vector = average_feature_vector(sentence, w2v)
            sentence_vectors.append(sentence_vector)
            feature_vector = np.add(feature_vector, sentence_vector)

        feature_vector = np.divide(feature_vector, len(features))

        total_similarity = 0
        for sentence_vector in sentence_vectors:
            total_similarity += 1 - spatial.distance.cosine(feature_vector, sentence_vector)

        avg_similarity[c] = total_similarity / len(sentence_vectors)
    return avg_similarity

def average_feature_vector(sentence, embedding):

    feature_vector = np.zeros((num_features, ), dtype="float32")

    for word in sentence:
        if word in embedding.vocab:
            word_vector = embedding.word_vec(word)
        else:
            word_vector = np.random.uniform(-0.01, 0.01, num_features).astype("float32")
        feature_vector = np.add(feature_vector, word_vector)

    feature_vector = np.divide(feature_vector, len(sentence))
    return feature_vector

data = utils.read_TREC()
avg_dist = average_distance_per_category(data["train_x"], data["train_y"], sorted(list(set(data["train_y"]))))

avg_dist = sorted([(key, value) for key, value in avg_dist.items()], key=lambda x: x[1])

pprint(avg_dist)
