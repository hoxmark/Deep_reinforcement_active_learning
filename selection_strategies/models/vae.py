from __future__ import print_function
import argparse
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from config import params, data

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(params["MAX_SENT_LEN"], 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, params["MAX_SENT_LEN"])

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # print(logvar)
        # print(mu)
        # print(self.training)
        # if self.training:
        #     print(logvar)
        #     std = logvar.mul(0.5).exp_()
        #     eps = Variable(std.data.new(std.size()).normal_())
        #     # print(eps)
        #     # print(eps)
        #     print(std)
        #     a = eps.mul(std)
        #     # print(a)
        #     # a = eps.mul(std).add_(mu)
        #     # print(a)
        #     return a
        # else:
        #     # print(mu)
            # print(mu)
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        # print(h3)
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 59))
        z = self.reparameterize(mu, logvar)
        # print(z)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # print(recon_x)
    # print(x)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 59))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    # KLD /= params["BATCH_SIZE"] * 59
    KLD = 0

    return BCE + KLD


def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(params["EPOCH"]):
        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] *
                       (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]

            model.train()
            train_loss = 0

            feature = Variable(torch.FloatTensor(batch_x))
            if params["CUDA"]:
                feature = feature.cuda()

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(feature)
            # print(recon_batch)
            loss = loss_function(recon_batch, feature, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch,loss.data[0] / len(feature)))


def test(model):
    model.eval()
    test_loss = 0
    for i in range(0, len(data["test_x"]), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(data["test_x"]) - i)

        batch_x = [[data["word_to_idx"][w] for w in sent] +
                   [params["VOCAB_SIZE"] + 1] *
                   (params["MAX_SENT_LEN"] - len(sent))
                   for sent in data["test_x"][i:i + batch_range]]

        data = Variable(torch.FloatTensor(batch_x), volatile=True)

        if params["CUDA"]:
            data = data.cuda()


        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

    test_loss /= len(data["text_x"])
    print('====> Test set loss: {:.4f}'.format(test_loss))
