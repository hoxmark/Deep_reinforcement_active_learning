import numpy as np
import sys
import random
from models.cnndqn import CNNDQN
import torch
import torch.nn as nn

class Game:
    def __init__(self, data, params):
        print("Initilizing the game:")
        self.params = params
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
        self.feature_extractor = CNNDQN(data, params)
        print("Story: length = ", len(self.train_x))
        self.order = list(range(0, len(self.train_x)))
        self.budget = params["BUDGET"]

        self.queried_times = 0
        self.queried_set_x = []
        self.queried_set_y = []
        self.queried_set_idx = []
        self.current_frame = 0
        self.terminal = False
        self.make_query = False
        self.performance = 0

    def get_frame(self, model):
        self.make_query = False
        sentence = self.train_x[self.order[self.current_frame]]
        sentence = torch.autograd.Variable(torch.LongTensor(sentence).unsqueeze(0))
        if self.params["CUDA"]:
            sentence = sentence.cuda()
        sentence_embedding = self.feature_extractor(sentence)
        entropy = self.calculate_entropy(model, sentence).unsqueeze(1)
        observation = torch.cat((sentence_embedding,entropy), dim=-1)
        return observation

    # tagger = crf model
    def feedback(self, action, model):
        reward = 0.
        is_terminal = False

        if action == 1:
            self.query()
            new_performance = self.get_performance(model)
            reward = new_performance - self.performance
            self.performance = new_performance
            # else:
                #reward = -1.
        else:
            reward = 0.

        # next frame
        next_sentence = []
        if self.queried_times == self.budget:
            # Return terminal
            return None, None, True

        self.current_frame += 1
        next_sentence = self.train_x[self.order[self.current_frame]]
        next_sentence = torch.autograd.Variable(torch.LongTensor(next_sentence).unsqueeze(0))
        if self.params["CUDA"]:
            next_sentence = next_sentence.cuda()

        next_sentence_embedding = self.feature_extractor(next_sentence)
        confidence = 0.
        entropy = self.calculate_entropy(model, next_sentence).unsqueeze(1)
        next_observation = torch.cat((next_sentence_embedding, entropy), dim=-1)
        print("Reward: {} - accuracy {}".format(reward, self.performance))
        return reward, next_observation, is_terminal

    def calculate_entropy(self, model, feature):
        output = model(feature)
        output = nn.functional.softmax(output)
        output = torch.mul(output, torch.log(output))
        output = torch.sum(output, dim=1)
        output = output * -1
        return output

    def query(self):
        sentence = self.train_x[self.order[self.current_frame]]
        # simulate: obtain the label
        label = self.train_y[self.order[self.current_frame]]
        self.queried_times += 1
        # print "Select:", sentence, label
        self.queried_set_x.append(sentence)
        self.queried_set_y.append(label)
        print("> Queried times", len(self.queried_set_x))

    # tagger = model
    def get_performance(self, model):
        # print(len(self.queried_set_x), len(self.queried_set_y))
        # model.init_model()
        model.train_model(self.queried_set_x, self.queried_set_y)
        performance = model.test(self.test_x, self.test_y)
        return performance
        # return 13.37
    #
    def reboot(self):
        random.shuffle(self.order)
        self.queried_times = 0
        self.terminal = False
        self.queried_set_x = []
        self.queried_set_y = []
        self.current_frame = 0
