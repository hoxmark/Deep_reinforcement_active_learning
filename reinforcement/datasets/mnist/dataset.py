from sklearn import datasets, svm, metrics
import sklearn
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import time


from config import opt 

def load_data():
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

        train_set = dset.MNIST(root=opt.data_path, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=opt.data_path , train=False, transform=trans, download=True)

        split = len(test_set)

        ##Train dataset        
        x = train_set.train_data.cpu().numpy()
        x = np.ravel(x)
        y = train_set.train_labels.cpu().numpy()
        
        ##Dev dataset
        x_dev = test_set.test_data.cpu().numpy()[:split]
        x_dev = np.ravel(x_dev)
        y_dev = test_set.test_labels.cpu().numpy()[:split]
        
        ##Test dataset
        x_test = test_set.test_data.cpu().numpy()[split:]
        x_test = np.ravel(x_test)
        y_test = test_set.test_labels.cpu().numpy()[split:]

        train_data = (x,y)
        dev_data = (x_dev, y_dev)
        test_data = (x_test, y_test)
        
        opt.state_size = len(set(y))
        opt.data_len = len(x)
        
        return train_data, dev_data, test_data