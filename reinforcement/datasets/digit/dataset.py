from sklearn import datasets, svm, metrics
import sklearn

from config import opt
# def load_data():
#         # The digits dataset
#         iris = datasets.load_iris()
#         # print(iris.)
#         n_samples = len(iris.data)
#         x = iris.data.reshape((n_samples, -1))
#         y = iris.target

#         x, y = sklearn.utils.shuffle(x, y)

#         dev_idx = n_samples // 10 * 8
#         test_idx = n_samples // 10 * 9

#         train_data = (x[:dev_idx], y[:dev_idx])
#         dev_data = (x[dev_idx:test_idx], y[dev_idx:test_idx])
#         test_data = (x[test_idx:], y[test_idx:])
#         opt.state_size = 10
#         opt.data_len = len(train_data[0])
#         return train_data, dev_data, test_data

def load_data():
        # The digits dataset
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        x = digits.images.reshape((n_samples, -1))
        y = digits.target

        x, y = sklearn.utils.shuffle(x, y)

        dev_idx = n_samples // 10 * 5
        test_idx = n_samples // 10 * 7

        train_data = (x[:dev_idx], y[:dev_idx])
        dev_data = (x[dev_idx:test_idx], y[dev_idx:test_idx])
        test_data = (x[test_idx:], y[test_idx:])
        # opt.state_size = len(x[0]) + 10
        # opt.data_size = len(x[0])
        # opt.pred_size = 10
        opt.data_sizes = [len(x[0]), 10]
        # print(opt.state_size)
        opt.data_len = len(train_data[0])
        return train_data, dev_data, test_data
