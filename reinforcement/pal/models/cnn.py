import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from sklearn.utils import shuffle

from gensim.models.keyedvectors import KeyedVectors
from config import params, data, w2v


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.BATCH_SIZE = params["BATCH_SIZE"]
        # self.SELECTION_SIZE = params["SELECTION_SIZE"]
        self.MAX_SENT_LEN = params["MAX_SENT_LEN"]
        self.WORD_DIM = params["WORD_DIM"]
        self.VOCAB_SIZE = params["VOCAB_SIZE"]
        self.CLASS_SIZE = params["CLASS_SIZE"]
        self.FILTERS = params["FILTERS"]
        self.FILTER_NUM = params["FILTER_NUM"]
        self.DROPOUT_EMBED_PROB = params["DROPOUT_EMBED"]
        self.DROPOUT_MODEL_PROB = params["DROPOUT_MODEL"]
        self.IN_CHANNEL = 1
        self.EMBEDDING = params["EMBEDDING"]


        # one for UNK and one for zero padding
        self.NUM_EMBEDDINGS = self.VOCAB_SIZE + 2
        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        if self.EMBEDDING != "random":
            self.wv_matrix = w2v["w2v"]

        self.init_model()

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))


    def init_model(self):
        self.embed = nn.Embedding(
            self.NUM_EMBEDDINGS, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)

        if self.EMBEDDING != "random":
            self.embed.weight.data.copy_(torch.from_numpy(self.wv_matrix))

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(
                self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, 'conv_{}'.format(i), conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        self.softmax = nn.Softmax()
        self.dropout_embed = nn.Dropout(self.DROPOUT_EMBED_PROB)
        self.dropout = nn.Dropout(self.DROPOUT_MODEL_PROB)

        if params["CUDA"]:
            self.cuda()

    def forward(self, inp):
        # inp = (25 x 59) - (mini_batch_size x sentence_length)
        x = self.embed(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        x = self.dropout_embed(x)
        # x = (25 x 1 x 17700) - mini_batch_size x embedding_for_each_sentence

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)),
                         self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]
        # Take a max for each filter - each filter result is 25 x 100 x 57

        # Each conv_result is (25 x 100)  - one max value for each application of each filter type, across each sentence
        x = torch.cat(conv_results, 1)
        # x = (25 x 300) - concatenate all the filter results
        x = self.dropout(x)
        x = self.fc(x)


        return x

    def test(self, test_x, test_y):
        self.eval()

        if params["CUDA"]:
            self.cuda()

        corrects, avg_loss = 0, 0
        for i in range(0, len(test_x), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(test_x) - i)

            feature = test_x[i : i + batch_range]
            target = test_y[i : i + batch_range]

            feature = Variable(torch.LongTensor(feature))
            target = Variable(torch.LongTensor(target))
            if params["CUDA"]:
                feature = feature.cuda()
                target = target.cuda()

            logit = self(feature)
            loss = torch.nn.functional.cross_entropy(logit, target, size_average=False)
            # avg_loss += loss.data[0]
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        size = len(test_x)
        # avg_loss = avg_loss / size
        accuracy = 100.0 * corrects / size

        return accuracy
