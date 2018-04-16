import numpy as np
import sys
import time
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from config import opt, data, loaders, global_logger
from evaluation import encode_data, i2t, t2i
from dataset import get_active_loader


class Game:
    def __init__(self):
        self.lg = global_logger["lg"]
        # self.reboot()

    def reboot(self, model):
        # self.order = random.sample(list(range(0, len(data["images_embed_all"]))), opt.budget)
        # self.load_episode_data()
        self.encode_episode_data(model)
        self.order = random.sample(list(range(0, len(data["images_embed_all"]))), len(data["images_embed_all"]))
        self.budget = opt.budget
        self.queried_times = 0
        self.current_state = 0
        self.performance = 0
        self.num_of_performances = 0
        # TODO delete data in loaders["active_loader"]

    # def load_episode_data(self):
    #     loaders["episode_loader"] = get_active_loader(opt.vocab)
    #     for index in self.order:
    #         loaders["episode_loader"].dataset.add_single(loaders["train_loader"].dataset[index][0],
    #                                                      loaders["train_loader"].dataset[index][1])

    # def encode_episode_data(self, model):
    #     time1 = time.time()
    #     img_embs, cap_embs = encode_data(model, loaders["episode_loader"])
    #     data["images_embed_episode"] = img_embs
    #     data["captions_embed_episode"] = cap_embs
    #     time2 = time.time()
    #
    #     print("Embeddings computed in {} ms".format((time2 - time1) * 1000.0))

    def encode_episode_data(self, model):
        time1 = time.time()
        img_embs, cap_embs = encode_data(model, loaders["train_loader"])
        data["images_embed_all"] = img_embs
        data["captions_embed_all"] = cap_embs
        time2 = time.time()

        print("Embeddings computed in {} ms".format((time2 - time1) * 1000.0))

    def get_state(self, model):
        time1 = time.time()
        image = torch.FloatTensor(data["images_embed_all"][self.order[self.current_state]]).unsqueeze(0)
        if opt.cuda:
            image = image.cuda()

        captions = data["captions_embed_all"]
        captions = torch.FloatTensor(captions)
        if opt.cuda:
            captions = captions.cuda()

        image_caption_distances = image.mm(captions.t())
        image_caption_distances_top10 = torch.abs(torch.topk(image_caption_distances, 10, 1, largest=False)[0])

        observation = torch.autograd.Variable(image_caption_distances_top10)
        if opt.cuda:
            observation = observation.cuda()
        self.current_state += 1
        time2 = time.time()
        print("State computed in {} ms".format((time2 - time1) * 1000.0))
        return observation

    def feedback(self, action, model):
        reward = 0.
        is_terminal = False

        if action == 1:
            self.query()
            new_performance = self.get_performance(model)
            reward = self.performance - new_performance
            self.performance = new_performance
        else:
            reward = 0.

        # TODO fix this
        if self.queried_times == self.budget:
            # Return terminal
            return None, None, True


        self.lg.scalar_summary("performance2", self.performance ,self.num_of_performances )
        self.num_of_performances += 1
        print("> State {:2} Action {:2} - reward {:4} - accuracy {:4}".format(self.current_state, action, reward, self.performance))
        next_observation = self.get_state(model)
        return reward, next_observation, is_terminal


    def query(self):
        index = self.order[self.current_state]
        image = loaders["train_loader"].dataset[index][0]
        caption = loaders["train_loader"].dataset[index][1]
        loaders["active_loader"].dataset.add_single(image, caption)

        self.queried_times += 1

    def get_performance(self, model):
        self.train_model(model, loaders["active_loader"])
        performance = self.validate(model)

        if (self.queried_times % 20 == 0):
            self.encode_episode_data(model)
        return performance

    def validate(self, model):
        # time1 = time.time()
        # # compute the encoding for all the validation images and captions
        # val_loader = loaders["val_loader"]
        # img_embs, cap_embs = encode_data(model, val_loader)
        # # caption retrieval
        # # (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure=opt.measure)
        # # image retrieval
        # (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, measure=opt.measure)
        # # sum of recalls to be used for early stopping
        # # performance = r1 + r5 + r10 + r1i + r5i + r10i
        # performance = r1i + r5i + r10i
        # time2 = time.time()
        # print("Validate in {} ms".format((time2 - time1) * 1000.0))

        time1 = time.time()
        performance = self.validate_loss(model)
        time2 = time.time()
        print("Validate losswise in {} ms".format((time2 - time1) * 1000.0))
        return performance

    def validate_loss(self, model):
        total_loss = 0
        model.val_start()
        for i, (images, captions, lengths, ids) in enumerate(loaders["val_loader"]):
            img_emb, cap_emb = model.forward_emb(images, captions, lengths, volatile=True)
            loss = model.forward_loss(img_emb, cap_emb)
            total_loss += loss.data[0]
        return total_loss

    def train_model(self, model, train_loader):
        time1 = time.time()
        model.train_start()

        for epoch in range(opt.num_epochs):
            self.adjust_learning_rate(model.optimizer, epoch)

            for i, train_data in enumerate(train_loader):
                # Always reset to train mode, this is not the default behavior
                model.train_start()
                # Update the model
                model.train_emb(*train_data)
        time2 = time.time()
        print("Train_model in {} ms".format((time2 - time1) * 1000.0))

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR
           decayed by 10 every 30 epochs"""
        lr = opt.learning_rate_vse * (0.1 ** (epoch // opt.lr_update))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
