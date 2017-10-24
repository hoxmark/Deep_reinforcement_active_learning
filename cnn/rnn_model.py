import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

class RNN(nn.Module):
    def __init__(self, params, data):
        super(RNN, self).__init__()
        self.params = params
        self.data = data

        self.MODEL = params["MODEL"]
        self.BATCH_SIZE = params["BATCH_SIZE"]
        self.MAX_SENT_LEN = params["MAX_SENT_LEN"]
        self.WORD_DIM = params["WORD_DIM"]
        self.VOCAB_SIZE = params["VOCAB_SIZE"]
        self.CLASS_SIZE = params["CLASS_SIZE"]
        self.FILTERS = params["FILTERS"]
        self.FILTER_NUM = params["FILTER_NUM"]
        self.DROPOUT_PROB = params["DROPOUT_PROB"]
        self.IN_CHANNEL = 1


        self.input_size = self.WORD_DIM * self.MAX_SENT_LEN
        # self.input_size = self.WORD_DIM
        self.hidden_size = 1200
        self.hidden_layers = 2
        self.output_size = params["CLASS_SIZE"]
        self.NUM_EMBEDDINGS = self.VOCAB_SIZE + 2

        assert (len(self.FILTERS) == len(self.FILTER_NUM))
        self.embedding = nn.Embedding(self.NUM_EMBEDDINGS, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL != "rand":
            wv_matrix = self.load_word2vec()
            self.embedding.weight.data.copy_(torch.from_numpy(wv_matrix))
        # self.rnn = nn.RNN(self.input_size, self.hidden_size, self.hidden_layers)
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.hidden_layers)
        self.i2o = nn.Linear(self.hidden_size, self.CLASS_SIZE)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden):
        x = self.embedding(input).view(-1, self.WORD_DIM * self.MAX_SENT_LEN)
        x = x.unsqueeze(1)
        output, hidden = self.rnn(x, hidden)
        output = output.squeeze(1)
        output = self.i2o(output)
        return output, hidden

    def init_hidden(self):
        # hidden = Variable(torch.zeros(self.hidden_layers, self.MAX_SENT_LEN, self.hidden_size))
        hidden = Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size))
        if self.params["CUDA"]:
            hidden = hidden.cuda(self.params["DEVICE"])
        return hidden


    """
    load word2vec pre trained vectors
    """
    def load_word2vec(self):
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format(
            "GoogleNews-vectors-negative300.bin", binary=True)

        wv_matrix = []
        for word in self.data["vocab"]:
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
