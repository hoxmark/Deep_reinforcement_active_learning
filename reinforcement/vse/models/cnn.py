import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from config import opt, data

from data.utils import timer


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.BATCH_SIZE = 32
        self.MAX_SENT_LEN = 59
        self.WORD_DIM = 300
        self.VOCAB_SIZE = 21425
        self.CLASS_SIZE = 2
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM = [100, 100, 100]
        self.DROPOUT_EMBED_PROB = 0.2
        self.DROPOUT_MODEL_PROB = 0.4
        self.IN_CHANNEL = 1

        # one for UNK and one for zero padding
        # self.NUM_EMBEDDINGS = self.VOCAB_SIZE + 2
        self.NUM_EMBEDDINGS = 21427
        assert (len(self.FILTERS) == len(self.FILTER_NUM))
        # self.wv_matrix = timer(self.load_word2vec, ())
        self.init_model()

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def init_model(self):
        self.embed = nn.Embedding(self.NUM_EMBEDDINGS, self.WORD_DIM, padding_idx=21425)

        # self.embed.weight.data.copy_(torch.from_numpy(self.wv_matrix))

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(
                self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, 'conv_{}'.format(i), conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        self.softmax = nn.LogSoftmax()
        self.dropout_embed = nn.Dropout(self.DROPOUT_EMBED_PROB)
        self.dropout = nn.Dropout(self.DROPOUT_MODEL_PROB)

        if opt.cuda:
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
        x = self.softmax(x)

        return x

    def get_sentence_representation(self, inp):
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
        # print(x)
        return x

    def load_word2vec(self):
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format(
            "{}/GoogleNews-vectors-negative300.bin".format(opt.data_path), binary=True)
        wv_matrix = []

        for word in data.vocab:
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(
                    np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        return wv_matrix

    def train_model(self, train_loader, epochs):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adadelta(parameters, 0.1)
        criterion = nn.NLLLoss()

        size = len(train_loader.dataset)

        if size > 0:
            self.train()
            for e in range(epochs):
                avg_loss = 0
                corrects = 0
                for i, train_data in enumerate(train_loader):
                    sentences, targets = train_data

                    features = Variable(sentences)
                    targets = Variable(targets)

                    if opt.cuda:
                        features, targets = features.cuda(), targets.cuda()

                    optimizer.zero_grad()
                    pred = self.forward(features)
                    loss = criterion(pred, targets)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.data[0]
                    corrects += (torch.max(pred, 1)
                                 [1].view(targets.size()).data == targets.data).sum()
                avg_loss = avg_loss * 32 / size

                if ((e + 1) % 10) == 0:
                    accuracy = 100.0 * corrects / size
                    dev_accuracy, dev_loss, dev_corrects, dev_size = evaluate(model, e, mode="dev")

                    s1 = "{:10s} loss: {:10.6f} acc: {:10.4f}%({}/{})".format(
                        "train", avg_loss, accuracy, corrects, size)
                    print(s1, end='\r')
                    # s2 = "{:10s} loss: {:10.6f} acc: {:10.4f}%({}/{})".format(
                        # "dev", dev_loss, dev_accuracy, dev_corrects, dev_size)
                    # output_lines[0] = s1

    def validate(self, loader):
        self.eval()
        corrects, avg_loss = 0, 0
        for i, data in enumerate(loader):
            feature, target = data

            feature = Variable(torch.LongTensor(feature))
            target = Variable(torch.LongTensor(target))
            feature.volatile = True

            if opt.cuda:
                feature = feature.cuda()
                target = target.cuda()

            logit = self.forward(feature)
            loss = torch.nn.functional.nll_loss(logit, target, size_average=False)
            avg_loss += loss.data[0]
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        size = len(loader.dataset)
        avg_loss = avg_loss / size
        accuracy = 100.0 * corrects / size

        # return accuracy, avg_loss, corrects, size
        return avg_loss
