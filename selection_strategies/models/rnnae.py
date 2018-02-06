
import os
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

from config import params, data

class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.hidden_size = params["HIDDEN_SIZE"]

        self.embedding = nn.Embedding(params["VOCAB_SIZE"] + 2, self.hidden_size, padding_idx=params["VOCAB_SIZE"] + 1)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input, batch_lengths):
        # TODO this makes first sentence include hidden states for padding .
        # find a fix for this 
        batch_lengths[0] = params["MAX_SENT_LEN"]
        output = self.embedding(input)

        output = torch.nn.utils.rnn.pack_padded_sequence(output, batch_lengths)
        hidden = self.initHidden(input.size()[1])
        output, hidden = self.gru(output, hidden)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if params["CUDA"]:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        self.hidden_size = params["HIDDEN_SIZE"]
        self.output_size = params["VOCAB_SIZE"] + 2

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, padding_idx=params["VOCAB_SIZE"] + 1)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).unsqueeze(0)
        output = F.relu(embedded)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = params["HIDDEN_SIZE"]
        self.output_size = params["VOCAB_SIZE"] + 2
        self.dropout_p = dropout_p
        self.max_length = params["MAX_SENT_LEN"]

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, padding_idx=params["VOCAB_SIZE"] + 1)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).unsqueeze(0)
        embedded = self.dropout(embedded)

        cat = torch.cat((embedded[0], hidden[0]), 1)
        temp = self.attn(cat)
        attn_weights = F.softmax(temp)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0, 1))

        output = torch.cat((embedded[0], attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


def train(encoder, decoder):
    print("Training autoencoder ")

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=params["LEARNING_RATE"])
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=params["LEARNING_RATE"])
    criterion = nn.NLLLoss()

    for epoch in range(params["EPOCH"]):
        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] for sent in data["train_x"][i:i + batch_range]]

            batch_x_sorted = sorted(batch_x, key=lambda x: len(x), reverse=True)

            batch_lengths = [len(sent) for sent in batch_x_sorted]
            # print(batch_lengths)

            batch_x_padded = [sent +[params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent)) for sent in batch_x_sorted]
            feature = Variable(torch.LongTensor(batch_x_padded))
            target = Variable(torch.LongTensor(batch_x_padded))

            feature, target = feature.transpose(0, 1), target.transpose(0, 1)
            if params["CUDA"]:
                feature, target = feature.cuda(), target.cuda()


            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(feature, batch_lengths)

            decoder_input = Variable(torch.LongTensor([params["VOCAB_SIZE"]] * len(batch_x)))
            if params["CUDA"]:
                decoder_input = decoder_input.cuda()

            decoder_hidden = encoder_hidden
            loss = 0

            # Teacher forcing: Feed the target as the next input
            all_decoded = torch.LongTensor()
            if params["CUDA"]:
                all_decoded = all_decoded.cuda()
            for di in range(params["MAX_SENT_LEN"]):
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target[di])
                decoder_input = target[di]  # Teacher forcing
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            print("{} of {}".format(i, len(data["train_x"])), end="\r")

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch, params["EPOCH"], loss.data[0] / len(batch_x)))
    print("Training finished")
