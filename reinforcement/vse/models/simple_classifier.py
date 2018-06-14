import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from config import opt, data

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.input_size = 64
        # TODO params
        self.hidden_size = 256
        self.output_size = 10

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inp):
        output = self.fc1(inp)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


    def train_model(self, loader, epochs):
        optimizer = optim.Adadelta(self.parameters(), 0.1)
        criterion = nn.CrossEntropyLoss()

        self.train()
        size = len(loader.dataset)

        for e in range(epochs):
            avg_loss = 0
            corrects = 0
            for i, (features, targets) in enumerate(loader):
                features, targets = Variable(features), Variable(targets)

                if opt.cuda:
                    features, targets = features.cuda(), targets.cuda()

                output = self.forward(features)
                optimizer.zero_grad()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                avg_loss += loss.data[0]
                corrects += (torch.max(output, 1)
                             [1].view(targets.size()).data == targets.data).sum()
            avg_loss = avg_loss / opt.batch_size
            # if ((e + 1) % 10) == 0:
            #     accuracy = 100.0 * corrects / size
            #     # dev_accuracy, dev_loss, dev_corrects, dev_size = evaluate(model, e, mode="dev")
            #
            #     s1 = "{:10s} loss: {:10.6f} acc: {:10.4f}%({}/{})".format(
            #         "train", avg_loss, accuracy, corrects, size)
            #     print(s1, end='\r')


    def predict_prob(self, inp):
        with torch.no_grad():
            output = self.forward(inp)
            output = torch.nn.functional.softmax(output, dim=1)
            return output

    def validate(self, loader):
        corrects, avg_loss = 0, 0
        with torch.no_grad():
            for i, data in enumerate(loader):
                feature, target = data

                feature = Variable(feature)
                target = Variable(target)

                if opt.cuda:
                    feature = feature.cuda()
                    target = target.cuda()

                logit = self.forward(feature)
                # loss = torch.nn.functional.nll_loss(logit, target, size_average=False)
                loss = torch.nn.functional.cross_entropy(logit, target, size_average=False)
                avg_loss += loss.data.item()
                corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

            size = len(loader.dataset)
            avg_loss = avg_loss / size
            accuracy = 100.0 * float(corrects) / float(size)

            metrics = {
                'accuracy': accuracy,
                'avg_loss': avg_loss,
                'performance': accuracy
            }

            return metrics

    def performance_validate(self, loader):
        return self.validate(loader)
