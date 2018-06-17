import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

from utils import pairwise_distances, batchify
from config import opt, data, loaders

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.input_size = 64
        # TODO params
        self.hidden_size = 256
        self.output_size = 10

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        self.reset()

        if opt.cuda:
            self.cuda()

    def reset(self):
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, inp):
        inp = Variable(inp)
        if opt.cuda:
            inp = inp.cuda()
        output = self.fc1(inp)
        output = self.relu(output)
        output = self.fc3(output)
        return output

    def train_model(self, data, epochs):
        optimizer = optim.Adadelta(self.parameters(), 0.1)
        criterion = nn.CrossEntropyLoss()

        self.train()
        size = len(data[0])

        for e in range(epochs):
            avg_loss = 0
            corrects = 0
            for i, (features, targets) in enumerate(batchify(data)):
                features = Variable(torch.FloatTensor(features))
                targets = Variable(torch.LongTensor(targets))

                if opt.cuda:
                    features, targets = features.cuda(), targets.cuda()

                output = self.forward(features)
                optimizer.zero_grad()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                avg_loss += loss.data.item()
                corrects += (torch.max(output, 1)
                             [1].view(targets.size()).data == targets.data).sum()
            avg_loss = avg_loss / opt.batch_size
            accuracy = 100.0 * corrects / size

            # s1 = "{:10s} loss: {:10.6f} acc: {:10.4f}%({}/{})".format("train", avg_loss, accuracy, corrects, size)
            # print(s1, end='\r')


    def predict_prob(self, inp):
        with torch.no_grad():
            output = self.forward(inp)
            output = torch.nn.functional.softmax(output, dim=1)
            return output

    def validate(self, data):
        corrects, avg_loss = 0, 0
        with torch.no_grad():
            for i, (features, targets) in enumerate(batchify(data)):
                features = Variable(torch.FloatTensor(features))
                targets = Variable(torch.LongTensor(targets))

                if opt.cuda:
                    features = features.cuda()
                    targets = targets.cuda()

                logit = self.forward(features)
                # loss = torch.nn.functional.nll_loss(logit, targets, size_average=False)
                loss = torch.nn.functional.cross_entropy(logit, targets, size_average=False)
                avg_loss += loss.data.item()
                corrects += (torch.max(logit, 1)[1].view(targets.size()).data == targets.data).sum()

            size = len(data[0])
            avg_loss = avg_loss / size
            accuracy = 100.0 * float(corrects) / float(size)

            metrics = {
                'accuracy': accuracy,
                'avg_loss': avg_loss,
                'performance': accuracy
            }

            return metrics

    def performance_validate(self, data):
        return self.validate(data)

    def get_state(self, index):
        state = data["all_predictions"][index]
        return state

    def encode_episode_data(self):
        images = []
        # for i, (features, targets) in enumerate(loaders["train_loader"]):
        for i, (features, targets) in enumerate(batchify(data["train_deleted"])):
            features = Variable(torch.FloatTensor(features))
            targets = Variable(torch.LongTensor(targets))

            preds = self.predict_prob(features)
            images.append(preds)

        images = torch.cat(images, dim=0)
        data["all_predictions"] = images


    def find_avg_pred_entropy(self, model):
        images = []
        tot = 0
        length = len(data["all_predictions"])

        for i, pred in enumerate(data["all_predictions"]):
            tot += entropy(pred)

        avg = tot/length
        return avg

    def query(self, index):
        current_state = data["all_predictions"][index].view(1, -1)
        all_states = data["all_predictions"]
        current_all_dist = pairwise_distances(current_state, all_states)
        similar_indices = torch.topk(current_all_dist, opt.selection_radius, 1, largest=False)[1]
        similar_indices = similar_indices.data[0].cpu().numpy()
        for idx in similar_indices:
            self.add_index(idx)
        return similar_indices

    def add_index(self, index):
        image = data["train_deleted"][0][index]
        caption = data["train_deleted"][1][index]
        data["active"][0].append(image)
        data["active"][1].append(caption)
