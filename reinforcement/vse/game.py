import numpy as np
import sys
import time
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from config import opt, data, loaders
from data.utils import timer, average_vector, get_distance, pairwise_distances
from data.evaluation import encode_data, i2t, t2i
from data.dataset import get_active_loader


class Game:
    def reboot(self, model):
        """resets the Game Object, to make it ready for the next episode """

        loaders["active_loader"] = get_active_loader(opt.vocab)
        data_len = loaders["train_loader"].dataset.length
        self.order = random.sample(list(range(0, data_len)), data_len)
        self.budget = opt.budget
        self.queried_times = 0
        self.current_state = 0

        self.init_train_k_random(model, opt.init_samples)

        if opt.embedding != 'static':
            self.encode_episode_data(model, loaders["train_loader"])
        self.performance = self.validate(model)

    def encode_episode_data(self, model, loader):
        img_embs, cap_embs = timer(encode_data, (model, loader))
        images, captions = torch.FloatTensor(img_embs), torch.FloatTensor(cap_embs)

        image_caption_distances = pairwise_distances(images, captions)
        image_caption_distances_topk = torch.topk(image_caption_distances, opt.topk, 1, largest=False)[0]

        if opt.cuda:
            image_caption_distances_topk = image_caption_distances_topk.cuda()


        data["images_embed_all"] = img_embs
        data["captions_embed_all"] = cap_embs
        data["image_caption_distances_topk"] = image_caption_distances_topk
        data["img_embs_avg"] = average_vector(data["images_embed_all"])
        data["cap_embs_avg"] = average_vector(data["captions_embed_all"])

    def get_state(self, model):
        current_idx = self.order[self.current_state]
        image_topk = data["image_caption_distances_topk"][current_idx].view(1, -1)
        state = image_topk

        if opt.image_distance:
            image = torch.FloatTensor(data["images_embed_all"][current_idx]).view(1, -1)
            img_distance = get_distance(image, data["img_embs_avg"])
            image_dist_tensor = torch.FloatTensor([img_distance])

            if opt.cuda:
                image_dist_tensor = image_dist_tensor.cuda()
                state = torch.cat((state, image_dist_tensor.view(1,-1)), 1)

        observation = torch.autograd.Variable(state)
        if opt.cuda:
            observation = observation.cuda()
        self.current_state += 1
        return observation

    def feedback(self, action, model):
        reward = 0.
        is_terminal = False

        if action == 1:
            timer(self.query, ())
            new_performance = self.get_performance(model)
            reward = self.performance - new_performance
            self.performance = new_performance
        else:
            reward = 0.

        # TODO fix this
        if self.queried_times >= self.budget:
            # Return terminal
            return None, None, True

        print("> State {:2} Action {:2} - reward {:.4f} - accuracy {:.4f}".format(
            self.current_state, action, reward, self.performance))
        next_observation = timer(self.get_state, (model,))
        return reward, next_observation, is_terminal

    def query(self):
        current = self.order[self.current_state]
        current_dist_vector = data["image_caption_distances_topk"][current].view(1, -1)
        all_dist_vectors = data["image_caption_distances_topk"]
        current_all_dist = pairwise_distances(current_dist_vector, all_dist_vectors)
        similar_indices = torch.topk(current_all_dist, opt.selection_radius, 1, largest=False)[1]

        for index in similar_indices[0]:
            image = loaders["train_loader"].dataset[index][0]
            caption = loaders["train_loader"].dataset[index][1]
            loaders["active_loader"].dataset.add_single(image, caption)
            self.queried_times += 1

    def init_train_k_random(self, model, num_of_init_samples):
        for i in range(0, num_of_init_samples):
            current = self.order[(-1*(i + 1))]
            image = loaders["train_loader"].dataset[current][0]
            caption = loaders["train_loader"].dataset[current][1]
            loaders["active_loader"].dataset.add_single(image, caption)

        # TODO: delete used init samples (?)
        timer(self.train_model, (model, loaders["active_loader"], 30))
        print("Validation after training on random data: {}".format(self.validate(model)))

    def get_performance(self, model):
        timer(self.train_model, (model, loaders["active_loader"]))
        performance = self.validate(model)

        if (self.queried_times % 20 == 0):
            if opt.embedding != 'static':
                self.encode_episode_data(model, loaders["train_loader"])
        return performance

    def performance_validate(self, model):
        """returns the performance messure with recall at 1, 5, 10
        for both image -> caption and cap -> img, and the sum of them all added together"""
        # compute the encoding for all the validation images and captions
        val_loader = loaders["val_tot_loader"]
        img_embs, cap_embs = encode_data(model, val_loader)
        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure=opt.measure)
        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, measure=opt.measure)

        performance = r1 + r5 + r10 + r1i + r5i + r10i
        return (performance, r1, r5, r10, r1i, r5i, r10i)

    def validate(self, model):
        performance = timer(self.validate_loss, (model,))
        return performance

    def validate_loss(self, model):
        total_loss = 0
        model.val_start()
        for i, (images, captions, lengths, ids) in enumerate(loaders["val_loader"]):
            img_emb, cap_emb = model.forward_emb(images, captions, lengths, volatile=True)
            loss = model.forward_loss(img_emb, cap_emb)
            total_loss += loss.data[0]
        return total_loss / loaders["val_loader"].dataset.length

    def train_model(self, model, train_loader, epochs=opt.num_epochs):
        model.train_start()
        if len(train_loader) > 0:
            for epoch in range(epochs):
                self.adjust_learning_rate(model.optimizer, epoch)
                for i, train_data in enumerate(train_loader):
                    model.train_start()
                    model.train_emb(*train_data)

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR
           decayed by 10 every 30 epochs"""
        lr = opt.learning_rate_vse * (0.1 ** (epoch // opt.lr_update))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
