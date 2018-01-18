import numpy as np
import sys
import random
from models.cnndqn import CNNDQN
# import helpers
#from gensim.models import doc2vec, word2vec
import torch
import torch.nn as nn


class Game:

    def __init__(self, data, params):
        # build environment
        # load data as story
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
        # self.w2v = w2v

        print("Story: length = ", len(self.train_x))
        self.order = list(range(0, len(self.train_x)))
        # if re-order, use random.shuffle(self.order)
        # load word embeddings, pretrained - w2v
        # print "Dictionary size", len(self.w2v), "Embedding size",
        # len(self.w2v[0])

        # when queried times is 100, then stop
        self.budget = params["BUDGET"]
        self.queried_times = 0

        # select pool
        self.queried_set_x = []
        self.queried_set_y = []
        self.queried_set_idx = []

        # let's start
        # self.episode = 0
        # story frame
        self.current_frame = 0
        #self.nextFrame = self.current_frame + 1
        self.terminal = False
        self.make_query = False
        self.performance = 0

    def get_frame(self, model):
        self.make_query = False
        sentence = self.train_x[self.order[self.current_frame]]
        # print(sentence)
        # sentence = torch.FloatTensor(sentence).unsqueeze(0)
        sentence = torch.autograd.Variable(torch.LongTensor(sentence).unsqueeze(0))
        if self.params["CUDA"]:
            sentence = sentence.cuda()
        # sentence =
        sentence_embedding = self.feature_extractor(sentence)
        # print(sentence_embedding)

        # confidence = 0.
        # predictions = []
        # if model.name == "CRF":
        #     confidence = model.get_confidence(sentence)
        #     predictions = model.get_predictions(sentence)
        # else:
        #     confidence = model.get_confidence(sentence_idx)
        #     predictions = model.get_predictions(sentence_idx)
        # preds_padding = []
        # orig_len = len(predictions)
        # if orig_len < self.max_len:
        #     preds_padding.extend(predictions)
        #     for i in range(self.max_len - orig_len):
        #         preds_padding.append([0] * 5)
        # elif orig_len > self.max_len:
        #     preds_padding = predictions[0:self.max_len]
        # else:
        #     preds_padding = predictions

        # Entropy
        # output = model(sentence_embedding)
        # output = nn.functional.softmax(output)
        # output = torch.mul(output, torch.log(output))
        # output = torch.sum(output, dim=1)
        # entorpy = output * -
        entropy = self.calculate_entropy(model, sentence).unsqueeze(1)
        # print(sentence_embedding)
        # print(entropy)

        # print(sentence_embedding)
        # print(entropy)
        observation = torch.cat((sentence_embedding,entropy), dim=-1)
        # torch.a
        # print(observation)

        # observation = torch.FloatTensor([sentence_embedding, entropy])
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
        # predictions = []
        # if model.name == "CRF":
        #     confidence = model.get_confidence(next_sentence)
        #     predictions = model.get_predictions(next_sentence)
        # else:
        #     confidence = model.get_confidence(next_sentence_idx)
        #     predictions = model.get_predictions(next_sentence_idx)
        # preds_padding = []
        # orig_len = len(predictions)
        # if orig_len < self.max_len:
        #     preds_padding.extend(predictions)
        #     for i in range(self.max_len - orig_len):
        #         preds_padding.append([0] * 5)
        # elif orig_len > self.max_len:
        #     preds_padding = predictions[0:self.max_len]
        # else:
        #     preds_padding = predictions

        # next_observation = [next_sentence, confidence, preds_padding]
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
        # train with {queried_set_x, queried_set_y}
        # train with examples: self.model.train(self.queried_set_x,
        # self.queried_set_y)
        print(len(self.queried_set_x), len(self.queried_set_y))

        # print train_sents
        model.init_model()
        model.train_model(self.queried_set_x, self.queried_set_y)
        # test on development data
        performance = model.test(self.test_x, self.test_y)
        #performance = self.model.test2conlleval(self.dev_x, self.dev_y)
        return performance
    #
    def reboot(self):
        # resort story
        # why not use docvecs? TypeError: 'DocvecsArray' object does not
        # support item assignment
        random.shuffle(self.order)
        self.queried_times = 0
        self.terminal = False
        self.queried_set_x = []
        self.queried_set_y = []
        self.current_frame = 0
        # self.episode += 1
        # print "> Next episode", self.episode
