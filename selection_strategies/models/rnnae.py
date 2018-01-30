
import os

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.optim as optim

from config import params, data

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

# input_size = params["VOCAB_SIZE"] + 2
# output_size = params["VOCAB_SIZE"] + 2
hidden_size = 256
max_length = 10


class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()

        self.hidden_size = 256

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Embedding(params["VOCAB_SIZE"] + 100, hidden_size, padding_idx=params["VOCAB_SIZE"] + 1)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # print(input)
        # print(params["VOCAB_SIZE"])
        test = self.embedding(input)
        # print(self.embedding(input).size())
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if params["CUDA"]:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = params["VOCAB_SIZE"] + 100

        self.embedding = nn.Embedding(self.output_size, hidden_size, padding_idx=params["VOCAB_SIZE"] + 1)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = params["VOCAB_SIZE"] + 100
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(params["VOCAB_SIZE"] + 100, hidden_size, padding_idx=params["VOCAB_SIZE"] + 1)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


def train(encoder, decoder):
    print("Training autoencoder ")

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    print(params["EPOCH"])

    for epoch in range(params["EPOCH"]):
        print(epoch)
        for i in range(0, len(data["train_x"])):
            sent = data["train_x"][i]
            if len(sent) < max_length:
                batch_x = [data["word_to_idx"][w] for w in sent]
                batch_x += [params["VOCAB_SIZE"] + 1 * (params["MAX_SENT_LEN"] - len(sent))]

                # batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)
                #
                # batch_x = [[data["word_to_idx"][w] for w in sent] +
                #            [params["VOCAB_SIZE"] + 1] *
                #            (params["MAX_SENT_LEN"] - len(sent))
                #            for sent in data["train_x"][i:i + batch_range]]


                feature = Variable(torch.LongTensor(batch_x))
                target = Variable(torch.LongTensor(batch_x))

                if params["CUDA"]:
                    feature, target = feature.cuda(), target.cuda()

                encoder_hidden = encoder.initHidden()

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                input_length = feature.size()[0]
                target_length = target.size()[0]

                encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))

                if params["CUDA"]:
                    encoder_outputs = encoder_outputs.cuda()

                loss = 0
                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        feature[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0][0]

                # SOS token = start of sentence token
                decoder_input = Variable(torch.LongTensor([[len(data["vocab"])]]))

                if params["CUDA"]:
                    decoder_input = decoder_input.cuda()

                decoder_hidden = encoder_hidden

                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)

                    # print(decoder_output)
                    # print(target[di])
                    loss += criterion(decoder_output, target[di])
                    decoder_input = target[di]  # Teacher forcing

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                # return


                print('epoch [{}/{}], loss:{:.4f}'.format(epoch, params["EPOCH"], loss.data[0] / target_length))
    print("Training finished")
