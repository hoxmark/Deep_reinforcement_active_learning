import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from config import opt, data

from data.utils import timer
from sklearn import datasets, svm, metrics


class SVM():
    def __init__(self):
        self.init_model()
        pass

    def init_model(self):
        self.classifier = svm.SVC(gamma=0.001, probability=True)
        

    def get_sentence_representation(self, inp):
        pass


    def train_model(self, loader, epochs):     
        size = len(loader.dataset)
        images = []
        targets = []
        if size > 0:          
            for i, train_data in enumerate(loader.dataset):                            
                
                images.append(train_data[0])
                targets.append(train_data[1])                
            self.classifier.fit(images, targets)

    def validate(self, loader):
        images = []
        targets = []
        size = len(loader.dataset)        
        for i, train_data in enumerate(loader.dataset):                                            
            images.append(train_data[0])
            targets.append(train_data[1])
        
        # Create a classifier: a support vector classifier
        expected = targets
        predicted = self.classifier.predict(images)
        # print(metrics.classification_report(expected, predicted))
        accuracy = metrics.accuracy_score(expected, predicted)
        return accuracy

    def performance_validate(self, loader):
        return self.validate(loader)
    

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
        # print('probs: {0}'.format(probs))
        # print('proba_ordered: {0}'.format(proba_ordered))
        
        # assert(len(probs)==10)
        
        return proba_ordered