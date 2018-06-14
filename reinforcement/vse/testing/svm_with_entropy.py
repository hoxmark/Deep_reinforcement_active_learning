# Standard scientific Python imports
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from utils import test_local_logger


entropy = True
lg =  test_local_logger(entropy)


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.utils import shuffle

# The digits dataset
digits = datasets.load_digits()
batch_size = 2

def selectRandomData(test_data, test_activeTargets, step): 

    return shuffle(test_data, test_activeTagets)

def selectBestData(test_data, test_activeTargets, step):
    total = 0
    if step==0: 
        return shuffle(test_data, test_activeTagets)

    
    probs = classifier.predict_proba(test_data)
    
    output = np.multiply(probs, np.log2(probs))
    output = np.sum(output, axis=1)
    output = output * -1
    test_data, test_activeTagets

    next_data = []
    next_taget = []
    order = np.argsort(-output)
    
    return np.array(test_data)[order], np.array(test_activeTagets)[order]


total = {}
num_of_iterations = 10

for iteration in range(0,num_of_iterations):
    
    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    test_data = data[:n_samples // 2]
    test_activeTagets = digits.target[:n_samples // 2]
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001, probability=True)
    
    # We learn the digits on the first half of the digits

    activeData =[]
    activeTargets =[]
    
    for step in range(0,50):   
        
        if entropy:        
            test_data, test_activeTagets = selectBestData(test_data, test_activeTagets, step)
        else: 
            test_data, test_activeTagets = selectRandomData(test_data, test_activeTagets, step)
        
        if step == 0:
            to_extract = batch_size*50
        else: 
            to_extract = batch_size
        activeData.extend(test_data[:to_extract])
        activeTargets.extend(test_activeTagets[:to_extract])
        test_data = test_data[to_extract:]
        test_activeTagets = test_activeTagets[to_extract:]
        
        classifier.fit(activeData, activeTargets)
        print(len(test_data))
        print(len(activeData))
        # # Now predict the value of the digit on the second half:
        expected = digits.target[n_samples // 2:]
        predicted = classifier.predict(data[n_samples // 2:])
        # print(metrics.classification_report(expected, predicted))
        acc = metrics.accuracy_score(expected, predicted)
        print(acc)
        if not step in total:
            total[step] = acc
        else:
            total[step] += acc
        lg.scalar_summary('acc/{}'.format(iteration), acc, step)

for k, v in enumerate(total):
    print(v)
    print(k)
    print(total[v])
    lg.scalar_summary('avg', total[v]/num_of_iterations, k)
