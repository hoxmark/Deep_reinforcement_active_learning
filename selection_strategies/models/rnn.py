import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from config import params, data, w2v


class RNN(nn.Module):
    def __init__(self, params, data):
        super(RNN, self).__init__()
        self.params = params
        self.data = data

        self.BATCH_SIZE = params["BATCH_SIZE"]
        self.SELECTION_SIZE = params["SELECTION_SIZE"]
        self.MAX_SENT_LEN = params["MAX_SENT_LEN"]
        self.WORD_DIM = params["WORD_DIM"]
        self.VOCAB_SIZE = params["VOCAB_SIZE"]
        self.CLASS_SIZE = params["CLASS_SIZE"]
        self.FILTERS = params["FILTERS"]
        self.FILTER_NUM = params["FILTER_NUM"]
        self.DROPOUT_EMBED_PROB = params["DROPOUT_EMBED"]
        self.DROPOUT_MODEL_PROB = params["DROPOUT_MODEL"]
        self.EMBEDDING = params["EMBEDDING"]


        self.input_size = self.WORD_DIM
        self.hidden_size = params["HIDDEN_SIZE"]
        self.hidden_layers = params["HIDDEN_LAYERS"]
        self.output_size = params["CLASS_SIZE"]
        self.NUM_EMBEDDINGS = self.VOCAB_SIZE + 2

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        if self.EMBEDDING != "random":
            self.wv_matrix = w2v["w2v"]

        self.init_model()

    def init_model(self):
        self.embed = nn.Embedding(self.NUM_EMBEDDINGS, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.EMBEDDING != "random":
            self.embed.weight.data.copy_(torch.from_numpy(self.wv_matrix))
        self.bigru = nn.GRU(self.WORD_DIM, self.hidden_size, dropout=self.DROPOUT_MODEL_PROB, num_layers=self.hidden_layers, bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_size * 2, self.CLASS_SIZE)
        self.dropout = nn.Dropout(self.DROPOUT_EMBED_PROB)

        if self.params["CUDA"]:
            self.cuda()

    def forward(self, input):
        if len(input.size()) == 1:
            input = input.unsqueeze(0)
        hidden = self.init_hidden(self.hidden_layers, len(input))
        # print(hidden)

        input = input.transpose(0, 1)
        embed = self.embed(input)
        # print(embed)
        embed = self.dropout(embed)  # add this reduce the acc
        input = embed.view(len(input), embed.size(1), -1)
        gru_out, hidden = self.bigru(input, hidden)
        # gru_out = (59 x 25 x 2400)

        gru_out = gru_out.permute(1, 2, 0)
        # gru_out = (25 x 2400 x 59)

        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # gru_out = (25 x 2400)
        gru_out = F.relu(gru_out)
        y = self.hidden2label(gru_out)
        return y


    def init_hidden(self, num_layers, batch_size):
        hidden = Variable(torch.zeros(num_layers * 2, batch_size, self.hidden_size))
        if self.params["CUDA"]:
            hidden = hidden.cuda()
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
