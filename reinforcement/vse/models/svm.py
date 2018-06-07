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
        pass

    
    def init_model(self):
        self.classifier = svm.SVC(gamma=0.001, probability=True)
        

    def get_sentence_representation(self, inp):
        pass


    def train_model(self, loader, epochs):
        # loader.dataset.shuffle()
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
