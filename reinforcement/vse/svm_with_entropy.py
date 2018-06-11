# Standard scientific Python imports
import matplotlib.pyplot as plt
import math
import random
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()
batch_size = 2
# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def selectRandomData(test_data, test_activeTargets, step): 
    # test_data, test_activeTargets = unison_shuffled_copies(test_data, test_activeTargets)
    return test_data, test_activeTagets

def selectBestData(test_data, test_activeTargets, step):
    total = 0
    if step==0: 
        # test_data, test_activeTargets = unison_shuffled_copies(test_data, test_activeTargets)
        return test_data, test_activeTagets
    
    probs = classifier.predict_proba(test_data)
    
    output = np.multiply(probs, np.log2(probs))
    output = np.sum(output, axis=1)
    output = output * -1
    test_data, test_activeTagets

    next_data = []
    next_taget = []
    order = np.argsort(-output)
    
    return np.array(test_data)[order], np.array(test_activeTagets)[order]
    
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

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
for step in range(0,100):   
    # test_data, test_activeTagets = selectBestData(test_data, test_activeTagets, step)
    test_data, test_activeTagets = selectRandomData(test_data, test_activeTagets, step)
    activeData.extend(test_data[:batch_size])
    activeTargets.extend(test_activeTagets[:batch_size])
    test_data = test_data[batch_size:]
    test_activeTagets = test_activeTagets[batch_size:]
    
    classifier.fit(activeData, activeTargets)
    
    # # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])
    # print(metrics.classification_report(expected, predicted))
    print(metrics.accuracy_score(expected, predicted))
    

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

# plt.show()