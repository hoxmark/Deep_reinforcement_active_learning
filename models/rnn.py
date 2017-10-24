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

        self.BATCH_SIZE = params["BATCH_SIZE"]
        self.MAX_SENT_LEN = params["MAX_SENT_LEN"]
        self.WORD_DIM = params["WORD_DIM"]
        self.VOCAB_SIZE = params["VOCAB_SIZE"]
        self.CLASS_SIZE = params["CLASS_SIZE"]
        self.FILTERS = params["FILTERS"]
        self.FILTER_NUM = params["FILTER_NUM"]
        self.DROPOUT_PROB = params["DROPOUT_PROB"]
        self.IN_CHANNEL = 1


        self.input_size = self.WORD_DIM
        self.hidden_size = params["HIDDEN_SIZE"]
        self.hidden_layers = params["HIDDEN_LAYERS"]
        self.output_size = params["CLASS_SIZE"]
        self.NUM_EMBEDDINGS = self.VOCAB_SIZE + 2

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        self.embedding = nn.Embedding(self.NUM_EMBEDDINGS, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if params["EMBEDDING"] != "random":
            wv_matrix = self.load_word2vec()
            self.embedding.weight.data.copy_(torch.from_numpy(wv_matrix))

        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.hidden_layers)
        self.i2o = nn.Linear(self.MAX_SENT_LEN * self.hidden_size, self.CLASS_SIZE)
        self.softmax = nn.Softmax()

        self.init_hidden()

    def forward(self, input):
        x = self.embedding(input)
        output, hidden = self.rnn(x, self.hidden_state)
        output = output.view(-1, self.MAX_SENT_LEN * self.hidden_size)
        output = self.i2o(output)
        output = self.softmax(output)

        self.hidden_state = hidden
        return output

    def init_hidden(self):
        self.hidden_state = Variable(torch.zeros(self.hidden_layers, self.MAX_SENT_LEN, self.hidden_size))
        if self.params["CUDA"]:
            self.hidden_state = self.hidden_state.cuda(self.params["DEVICE"])


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
