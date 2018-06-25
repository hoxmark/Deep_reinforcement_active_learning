import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pprint import pprint
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
        if opt.cuda:
            inp = inp.cuda()
        output = self.fc1(inp)
        output = self.relu(output)
        output = self.fc3(output)
        return output

    def train_model(self, train_data, epochs):
        optimizer = optim.Adadelta(self.parameters(), 0.1)
        criterion = nn.CrossEntropyLoss()

        self.train()
        size = len(train_data[0])
        if size > 0:
            for e in range(epochs):
                avg_loss = 0
                corrects = 0
                for i, (features, targets) in enumerate(batchify(train_data)):
                    features = torch.FloatTensor(features)
                    targets = torch.LongTensor(targets)

                    if opt.cuda:
                        features, targets = features.cuda(), targets.cuda()

                    output = self.forward(features)
                    optimizer.zero_grad()
                    loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()
                    corrects += (torch.max(output, 1)
                                 [1].view(targets.size()) == targets).sum()
                avg_loss = avg_loss / opt.batch_size
                accuracy = 100.0 * corrects / size


    def predict_prob(self, inp):
        with torch.no_grad():
            output = self.forward(inp)
            output = torch.nn.functional.softmax(output, dim=1)
            return output

    def validate(self, data):
        corrects, avg_loss = 0, 0
        with torch.no_grad():
            for i, (features, targets) in enumerate(batchify(data)):
                features = torch.FloatTensor(features)
                targets = torch.LongTensor(targets)

                if opt.cuda:
                    features = features.cuda()
                    targets = targets.cuda()

                logit = self.forward(features)
                loss = torch.nn.functional.cross_entropy(logit, targets, size_average=False)
                avg_loss += loss.item()
                corrects += (torch.max(logit, 1)[1].view(targets.size()) == targets).sum()

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
        img = torch.Tensor(data["train"][0][index])
        if opt.cuda:
            img = img.cuda()
        preds = self.forward(img)
        state = torch.cat((img, preds)).view(1, -1)
        return state

    def encode_episode_data(self):
        pass
        # images = []
        # # for i, (features, targets) in enumerate(loaders["train_loader"]):
        # all_states = torch.Tensor(data["train"][0])
        # for i, (features, targets) in enumerate(batchify(data["train"])):
        #     features = Variable(torch.FloatTensor(features))
        #     preds = self.predict_prob(features)
        #     images.append(preds)
        #
        # images = torch.cat(images, dim=0)
        #
        # # data["all_predictions"] = images
        # data["all_states"] = torch.cat((all_states, images.cpu()), dim=1)


    def query(self, index):
        # current_state = data["all_states"][index].view(1, -1)
        # all_states = data["all_states"]
        # current_all_dist = pairwise_distances(current_state, all_states)
        # similar_indices = torch.topk(current_all_dist, opt.selection_radius, 1, largest=False)[1]
        # similar_indices = similar_indices.data[0].cpu().numpy()
        # for idx in similar_indices:
        self.add_index(index)
        return [index]

    def add_index(self, index):
        image = data["train"][0][index]
        caption = data["train"][1][index]
        data["active"][0].append(image)
        data["active"][1].append(caption)
