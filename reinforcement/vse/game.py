import numpy as np
import sys
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from config import opt, data, loaders
from evaluation import encode_data, i2t, t2i


class Game:
    def __init__(self):
        print("Initilizing the game:")
        print("Story: length = ", len(data["images"]))
        self.order = list(range(0, len(data["images"])))

        self.budget = opt.budget
        self.queried_times = 0
        self.current_state = 0
        self.performance = 0

    def get_state(self, model):
        image = torch.FloatTensor(data["images"][self.order[self.current_state]]).unsqueeze(0)
        if opt.cuda:
            image = image.cuda()


        captions = data["captions"]
        captions = torch.FloatTensor(captions)
        if opt.cuda:
            captions = captions.cuda()

        image_caption_distances = image.mm(captions.t())
        image_caption_distances_top10 = torch.abs(torch.topk(image_caption_distances, 10, 1, largest=False)[0])

        # observation = torch.autograd.Variable(torch.FloatTensor().unsqueeze(0))
        observation = torch.autograd.Variable(image_caption_distances_top10)
        if opt.cuda:
            observation = observation.cuda()

        self.current_state += 1
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

        # TODO fix this
        if self.queried_times == self.budget:
            # Return terminal
            return None, None, True

        print("> Action {:2} - reward {:4} - accuracy {:4}".format(action, reward, self.performance))
        next_observation = self.get_state(model)
        return reward, next_observation, is_terminal

    def calculate_entropy(self, model, feature):
        output = model(feature)
        output = nn.functional.softmax(output)
        output = torch.mul(output, torch.log(output))
        output = torch.sum(output, dim=1)
        output = output * -1
        return output.data[0]

    def query(self):
        index = self.order[self.current_state]
        image = loaders["train_loader"].dataset[index][0]
        caption = loaders["train_loader"].dataset[index][1]
        loaders["active_loader"].dataset.add_single(image, caption)

        self.queried_times += 1

    def get_performance(self, model):
        self.train_model(model, loaders["active_loader"])
        performance = self.validate(model)

        # Model is re-trained - compute the embeddings again
        self.compute_embeddings(model)
        return performance

    def compute_embeddings(self, model):
        img_embs, cap_embs = encode_data(model, loaders["train_loader"])
        data["images"] = img_embs
        data["captions"] = cap_embs

    def validate(self, model):
        # compute the encoding for all the validation images and captions
        val_loader = loaders["val_loader"]
        img_embs, cap_embs = encode_data(model, val_loader)
        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure=opt.measure)
        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, measure=opt.measure)
        # sum of recalls to be used for early stopping
        performance = r1 + r5 + r10 + r1i + r5i + r10i
        return performance

    def train_model(self, model, train_loader):
        model.train_start()

        for epoch in range(opt.num_epochs):
            self.adjust_learning_rate(model.optimizer, epoch)

            for i, train_data in enumerate(train_loader):
                # Always reset to train mode, this is not the default behavior
                model.train_start()
                # Update the model
                model.train_emb(*train_data)

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR
           decayed by 10 every 30 epochs"""
        lr = opt.learning_rate_vse * (0.1 ** (epoch // opt.lr_update))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def reboot(self):
        random.shuffle(self.order)
        self.queried_times = 0
        self.current_state = 0
        # TODO delete data in loaders["active_loader"]
