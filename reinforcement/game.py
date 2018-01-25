import numpy as np
import sys
import random
from models.cnndqn import CNNDQN
import torch
import torch.nn as nn
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
        print("CURRENT FRAME", self.current_frame)
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

        self.current_frame += 1
        next_observation = self.get_frame(model)
        print("Reward: {} - accuracy {}".format(reward, self.performance))
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
        # simulate: obtain the label
        label = self.train_y[self.order[self.current_frame]]
        self.queried_times += 1
        # print "Select:", sentence, label
        self.queried_set_x.append(sentence)
        self.queried_set_y.append(label)
        print("> Queried times", len(self.queried_set_x))

    def get_performance(self, model):
        # model.init_model()
        print(self.queried_set_x)
        model.train_model(self.queried_set_x, self.queried_set_y)
        performance = model.test(self.test_x, self.test_y)
        return performance

    def reboot(self):
        random.shuffle(self.order)
        self.queried_times = 0
        self.queried_set_x = []
        self.queried_set_y = []
        self.current_frame = 0
