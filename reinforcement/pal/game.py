import numpy as np
import sys
import random
from models.cnndqn import CNNDQN
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from config import params, data


class Game:
    def __init__(self):
        print("Initilizing the game:")
        self.train_x = [[data["word_to_idx"][w] for w in sent] +
                        [params["VOCAB_SIZE"] + 1] *
                        (params["MAX_SENT_LEN"] - len(sent))
                        for sent in data["train_x"]]

        self.train_y = [data["classes"].index(c) for c in data["train_y"]]

        self.dev_x = [[data["word_to_idx"][w] for w in sent] +
                      [params["VOCAB_SIZE"] + 1] *
                      (params["MAX_SENT_LEN"] - len(sent))
                      for sent in data["dev_x"]]

        self.dev_y = [data["classes"].index(c) for c in data["dev_y"]]

        self.test_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] *
                       (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["test_x"]]

        self.test_y = [data["classes"].index(c) for c in data["test_y"]]
        self.max_len = params["MAX_SENT_LEN"]
        self.feature_extractor = CNNDQN()
        print("Story: length = ", len(self.train_x))
        self.order = list(range(0, len(self.train_x)))

        self.budget = params["BUDGET"]
        self.queried_times = 0
        self.queried_set_x = []
        self.queried_set_y = []
        self.current_frame = 0
        self.performance = 0

    def get_frame(self, model):
        sentence = self.train_x[self.order[self.current_frame]]
        sentence = torch.autograd.Variable(torch.LongTensor(sentence).unsqueeze(0))
        if params["CUDA"]:
            sentence = sentence.cuda()

        entropy = self.calculate_entropy(model, sentence)
        predictions = nn.functional.softmax(model(sentence))
        predictions = torch.sort(predictions, dim=1, descending=True)
        margin = predictions[0].data[0][0] - predictions[0].data[0][1]

        observation = torch.autograd.Variable(torch.FloatTensor([margin, entropy]).unsqueeze(0))
        if params["CUDA"]:
            observation = observation.cuda()

        self.current_frame += 1
        return observation

    def feedback(self, action, model):
        reward = 0.
        is_terminal = False

        if action == 1:
            self.query()
            new_performance = self.get_performance(model)
            reward = new_performance - self.performance
            self.performance = new_performance
        else:
            reward = 0.

        if self.queried_times == self.budget:
            # Return terminal
            return None, None, True

        print("> Action {:2} - reward {:4} - accuracy {:4}".format(action, reward, self.performance))
        next_observation = self.get_frame(model)
        return reward, next_observation, is_terminal

    def calculate_entropy(self, model, feature):
        output = model(feature)
        output = nn.functional.softmax(output)
        output = torch.mul(output, torch.log(output))
        output = torch.sum(output, dim=1)
        output = output * -1
        return output.data[0]

    def query(self):
        sentence = self.train_x[self.order[self.current_frame]]
        label = self.train_y[self.order[self.current_frame]]
        self.queried_times += 1
        self.queried_set_x.append(sentence)
        self.queried_set_y.append(label)

    def get_performance(self, model):
        # model.init_model()
        self.train_model(model)
        performance = model.test(self.test_x, self.test_y)
        return performance


    def train_model(self, model):
        model.train()
        print("Training model. Training set contains {} elements".format(len(self.queried_set_x)))
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
        criterion = nn.CrossEntropyLoss()

        for e in range(params["EPOCH"]):
            for i in range(0, len(self.queried_set_x), params["BATCH_SIZE"]):
                batch_range = min(params["BATCH_SIZE"], len(self.queried_set_x) - i)
                batch_x = self.queried_set_x[i:i + batch_range]
                batch_y = self.queried_set_y[i:i + batch_range]

                feature = Variable(torch.LongTensor(batch_x))
                target = Variable(torch.LongTensor(batch_y))

                if params["CUDA"]:
                    feature, target = feature.cuda(params["DEVICE"]), target.cuda(params["DEVICE"])

                optimizer.zero_grad()
                pred = model(feature)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()

            print("{} of {}".format(e, params["EPOCH"]), end='\r')

    def reboot(self):
        random.shuffle(self.order)
        self.queried_times = 0
        self.queried_set_x = []
        self.queried_set_y = []
        self.current_frame = 0
