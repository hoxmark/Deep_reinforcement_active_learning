import numpy as np
import sys
import time
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from config import opt, data, loaders
from utils import timer, average_vector, get_distance, pairwise_distances
# from data.evaluation import encode_data, i2t, t2i
# from datasets.dataset import get_active_loader
from scipy.stats import entropy


class Game:
    def reboot(self, model):
        """resets the Game Object, to make it ready for the next episode """

        # loaders["active_loader"] = get_active_loader(opt.batch_size)
        # data_len = loaders["train_loader"].dataset.length-opt.init_samples
        # self.order = random.sample(list(range(0, data_len // 5)), data_len // 5)
        data_len = len(data["train"][0])
        data["active"] = ([], [])

        self.order = random.sample(list(range(0, data_len)), data_len)
        self.budget = opt.budget
        self.queried_times = 0
        self.current_state = 0
        self.entropy = 0

        self.init_train_k_random(model, opt.init_samples)
        model.encode_episode_data()
        # self.construct_all_predictions(model)
        # self.avg_entropy_in_train_loader = self.get_avg_entropy_in_train_loader(loaders["train_loader"])
        # if opt.embedding != 'static':
            # self.encode_episode_data(model, loaders["train_loader"])
        # metrics = model.validate(loaders["val_loader"])
        metrics = model.validate(data["dev"])
        self.performance = metrics["performance"]


    def init_train_k_random(self, model, num_samples):
        for i in range(0, num_samples):
            current = self.order[(-1*(i + 1))]
            # image = loaders["train_loader"].dataset[current][0]
            # caption = loaders["train_loader"].dataset[current][1]
            image = data["train"][0][current]
            caption = data["train"][1][current]
            data["active"][0].append(image)
            data["active"][1].append(caption)
            # loaders["active_loader"].dataset.add_single(image, caption)
        # TODO: delete used init samples (?)
        # timer(model.train_model, (loaders["active_loader"], 100))
        timer(model.train_model, (data["active"], 100))


    # def encode_episode_data(self, model, loader):
    #     img_embs, cap_embs = timer(model.encode_data, (loader,))
    #     captions = torch.FloatTensor(cap_embs)
    #     images = []
    #
    #     # TODO dynamic im_div
    #     for i in range(0, len(img_embs), 5):
    #         images.append(img_embs[i])
    #     images = torch.FloatTensor(images)
    #
    #     image_caption_distances = pairwise_distances(images, captions)
    #     image_caption_distances_topk = torch.topk(image_caption_distances, opt.topk, 1, largest=False)[0]
    #
    #     data["images_embed_all"] = images
    #     data["captions_embed_all"] = captions
    #     data["image_caption_distances_topk"] = image_caption_distances_topk
    #     # data["img_embs_avg"] = average_vector(data["images_embed_all"])
    #     # data["cap_embs_avg"] = average_vector(data["captions_embed_all"])

    def get_state(self, model):
        current_idx = self.order[self.current_state]
        state = model.get_state(current_idx)
        state = Variable(torch.FloatTensor(state).view(1, -1))
        state = state.sort(descending=True)[0]
        if opt.cuda:
            state = state.cuda()

        self.current_state += 1
        return state

    # def construct_distance_state(self, index):
    #     # Distances to topk closest captions
    #     image_topk = data["image_caption_distances_topk"][index].view(1, -1)
    #     state = image_topk
    #
    #     # Distances to topk closest images
    #     if opt.topk_image > 0:
    #         current_image = data["images_embed_all"][index].view(1 ,-1)
    #         all_images = data["images_embed_all"]
    #         image_image_dist = pairwise_distances(current_image, all_images)
    #         image_image_dist_topk = torch.topk(image_image_dist, opt.topk_image, 1, largest=False)[0]
    #
    #         state = torch.cat((state, image_image_dist_topk), 1)
    #
    #     # Distance from average image vector
    #     if opt.image_distance:
    #         current_image = data["images_embed_all"][index].view(1 ,-1)
    #         img_distance = get_distance(current_image, data["img_embs_avg"].view(1, -1))
    #         image_dist_tensor = torch.FloatTensor([img_distance]).view(1, -1)
    #         state = torch.cat((state, image_dist_tensor), 1)
    #
    #     observation = torch.autograd.Variable(state)
    #     if opt.cuda:
    #         observation = observation.cuda()
    #     return observation


    def feedback(self, action, model):
        reward = 0.
        is_terminal = False
        if action == 1:
            timer(self.query, (model,))
            new_performance = self.get_performance(model)
            reward = new_performance - self.performance - opt.reward_threshold
            # if opt.reward_clip:
                # reward = np.tanh(reward / 100)
            reward = new_performance - self.performance - 1
            self.performance = new_performance
        else:
            reward = 0.

        print("> State {:2} Action {:2} - reward {:.4f} - accuracy {:.4f}".format(
            self.current_state, action, reward, self.performance))
        next_observation = self.get_state(model)

        if self.queried_times >= self.budget or self.current_state >= len(self.order):
            # Return terminal
            return reward, next_observation, True

        return reward, next_observation, is_terminal

    def query(self, model):
# <<<<<<< HEAD
#         if (len(self.order) == self.current_state):
#             return True
#         current = self.order[self.current_state]
#         image = loaders["train_loader"].dataset[current][0]
#         self.entropy = entropy(loaders["train_loader"].dataset[current][0][0])
#
#         caption = loaders["train_loader"].dataset[current][1]
#
#         # There are 5 captions for every image
#         loaders["active_loader"].dataset.add_single(image, caption)
#
#         self.queried_times += 1
#
#         return False
    # def query(self, model):
    #     self.construct_all_predictions(model)
    #     if (len(self.order) == self.current_state):
    #         return True
    #     current = self.order[self.current_state]
    #     # construct_state = self.construct_entropy_state
    #     #
    #     # current_state = construct_state(model, current)
    #     # all_states = torch.cat([construct_state(model, index) for index in range(len(self.order))])

    #     current_state = data["all_predictions"][current]
    #     all_states = data["all_predictions"]
    #     current_state = torch.from_numpy(current_state).view(1,-1)
    #     all_states = torch.from_numpy(all_states)
    #     current_all_dist = pairwise_distances(current_state, all_states)
    #     similar_indices = torch.topk(current_all_dist, opt.selection_radius, 1, largest=False)[1]

    #     for index in similar_indices.cpu().numpy():
    #         image = loaders["train_loader"].dataset[index][0]

    #         self.entropy = entropy(loaders["train_loader"].dataset[index][0][0])
    #         caption = loaders["train_loader"].dataset[index][1]
    #         # There are 5 captions for every image
    #         loaders["active_loader"].dataset.add_single(image[0], caption[0])

    #         # Only count images as an actual request.
    #         # Reuslt is that we have 5 times as many training points as requests.
    #         self.queried_times += 1

    #     return False
        current = self.order[self.current_state]
        model.query(current)
        self.queried_times += opt.selection_radius

        # for index in similar_indices[0]:
        #     image = loaders["train_loader"].dataset[5 * index][0]
        #     # There are 5 captions for every image
        #     for cap in range(5):
        #         caption = loaders["train_loader"].dataset[5 * index + cap][1]
        #         loaders["active_loader"].dataset.add_single(image, caption)
        #     # Only count images as an actual request.
        #     # Reuslt is that we have 5 times as many training points as requests.
        #     self.queried_times += 1


    def get_performance(self, model):
        # timer(model.train_model, (loaders["active_loader"], opt.num_epochs))
        timer(model.train_model, (data["active"], opt.num_epochs))
        # metrics = model.validate(loaders["val_loader"])
        metrics = model.validate(data["dev"])
        performance = metrics["performance"]
        model.encode_episode_data()
        return performance
    #
    # def adjust_learning_rate(self, optimizer, epoch):
    #     """Sets the learning rate to the initial LR
    #        decayed by 10 every 30 epochs"""
    #     lr = opt.learning_rate_vse * (0.1 ** (epoch // opt.lr_update))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
