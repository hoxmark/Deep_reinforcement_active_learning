
import torch.optim as optim
from torch import nn, FloatTensor
from torch.autograd import Variable

from config import params, data

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(params["MAX_SENT_LEN"], 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 32))
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, params["MAX_SENT_LEN"]))

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


def train(model):
    print("Training autoencoder ")
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(params["EPOCH"]):
        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] *
                       (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]

            model.train()

            feature = Variable(FloatTensor(batch_x))

            if params["CUDA"]:
                feature = feature.cuda()
            output = model(feature)
            loss = criterion(output, feature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch, params["EPOCH"], loss.data[0] / len(batch_x)))
