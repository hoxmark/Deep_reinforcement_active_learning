import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from config import opt, data

from utils import timer, batchify, pairwise_distances
from sklearn import datasets, svm, metrics


class SVM():
    def __init__(self):
        # print("INITDED")
        self.init_model()

    def init_model(self):
        self.classifier = svm.SVC(gamma=0.001, probability=True)

    def get_sentence_representation(self, inp):
        pass

    def cuda(self):
        return self 

    def get_state(self, index):
        state = Variable(torch.FloatTensor(data["all_predictions"][index]))
        return state

    def train_model(self, d, epochs):
        images = []
        targets = []    
        for i, (feat, tar) in enumerate(batchify(d)):    
            images.extend(feat)
            targets.extend(tar)
        self.classifier.fit(images, targets)

    def validate(self, loader):
        images = []
        targets = []
        for i, (feat, tar) in enumerate(batchify(loader)):    
            images.extend(feat)
            targets.extend(tar)

        # Create a classifier: a support vector classifier
        expected = targets
        predicted = self.classifier.predict(images)
        # print(metrics.classification_report(expected, predicted))
        accuracy = metrics.accuracy_score(expected, predicted)

        return {
                'accuracy': accuracy,
                'performance': accuracy
            }
        

    def performance_validate(self, loader):
        return self.validate(loader)

    def encode_episode_data(self):
        images = []
        targets = []
        for i, (feat, tar) in enumerate(batchify(data["train"])):    
            images.extend(feat)
            targets.extend(tar)

        
        preds = self.predict_prob(images)

        data["all_predictions"] = preds

        

    def predict_proba_ordered(self, probs, classes_, all_classes):
        """
        probs: list of probabilities, output of predict_proba
        classes_: clf.classes_
        all_classes: all possible classes (superset of classes_)
        """
        proba_ordered = np.zeros((probs.shape[0], all_classes.size),  dtype=np.float)
        sorter = np.argsort(all_classes) # http://stackoverflow.com/a/32191125/395857
        idx = sorter[np.searchsorted(all_classes, classes_, sorter=sorter)]
        proba_ordered[:, idx] = probs
        return proba_ordered

    def predict_prob(self, images):
        all_classes = np.array([0,1,2,3,4,5,6,7,8,9]) # explicitly set the possible class labels.
        probs = self.classifier.predict_proba(images) #As label 3 isn't in train set, the probs' size is 3, not 4
        proba_ordered = self.predict_proba_ordered(probs, self.classifier.classes_, all_classes)
        # print(proba_ordered)
        # print('probs: {0}'.format(probs))
        # print('proba_ordered: {0}'.format(proba_ordered))

        # assert(len(probs)==10)

        return proba_ordered

    def query(self, index):
        data["active"][0].append(data["train"][0][index])
        data["active"][1].append(data["train"][1][index])
        # current_state = data["all_predictions"][index]
        # all_states = data["all_predictions"]
        # current_all_dist = pairwise_distances(current_state, all_states)
        # similar_indices = torch.topk(current_all_dist, opt.selection_radius, 1, largest=False)[1]

        # for idx in similar_indices.data[0].cpu().numpy():
        #     image = data["train"][0][idx]
        #     caption = data["train"][1][idx]
        #     # There are 5 captions for every image
        #     # loaders["active_loader"].dataset.add_single(image, caption)
        #     data["active"][0].append(image)
        #     data["active"][1].append(caption)
        #     # Only count images as an actual request.
        #     # Reuslt is that we have 5 times as many training points as requests.
        #     # self.queried_times += 1