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
# from data.evaluation import encode_data, i2t, t2i
from data.dataset import get_active_loader


class Game:
    def reboot(self, model):
        """resets the Game Object, to make it ready for the next episode """

        loaders["active_loader"] = get_active_loader(opt.batch_size)
        data_len = loaders["train_loader"].dataset.length
        # self.order = random.sample(list(range(0, data_len // 5)), data_len // 5)
        self.order = random.sample(list(range(0, data_len)), data_len)
        self.budget = opt.budget
        self.queried_times = 0
        self.current_state = 0

        self.init_train_k_random(model, opt.init_samples)

        if opt.embedding != 'static':
            self.encode_episode_data(model, loaders["train_loader"])
        metrics = model.validate(loaders["val_loader"])
        self.performance = metrics["avg_loss"]

    def encode_episode_data(self, model, loader):
        img_embs, cap_embs = timer(model.encode_data, (loader,))
        captions = torch.FloatTensor(cap_embs)
        images = []

        # TODO dynamic im_div
        for i in range(0, len(img_embs), 5):
            images.append(img_embs[i])
        images = torch.FloatTensor(images)

        image_caption_distances = pairwise_distances(images, captions)
        image_caption_distances_topk = torch.topk(image_caption_distances, opt.topk, 1, largest=False)[0]

        data["images_embed_all"] = images
        data["captions_embed_all"] = captions
        data["image_caption_distances_topk"] = image_caption_distances_topk
        data["img_embs_avg"] = average_vector(data["images_embed_all"])
        data["cap_embs_avg"] = average_vector(data["captions_embed_all"])

    def get_state(self, model):
        current_idx = self.order[self.current_state]
        # observation = self.construct_distance_state(current_idx)
        observation = self.construct_entropy_state(model, current_idx)
        self.current_state += 1
        return observation

    def construct_distance_state(self, index):
        # Distances to topk closest captions
        image_topk = data["image_caption_distances_topk"][index].view(1, -1)
        state = image_topk

        # Distances to topk closest images
        if opt.topk_image > 0:
            current_image = data["images_embed_all"][index].view(1 ,-1)
            all_images = data["images_embed_all"]
            image_image_dist = pairwise_distances(current_image, all_images)
            image_image_dist_topk = torch.topk(image_image_dist, opt.topk_image, 1, largest=False)[0]

            state = torch.cat((state, image_image_dist_topk), 1)

        # Distance from average image vector
        if opt.image_distance:
            current_image = data["images_embed_all"][index].view(1 ,-1)
            img_distance = get_distance(current_image, data["img_embs_avg"].view(1, -1))
            image_dist_tensor = torch.FloatTensor([img_distance]).view(1, -1)
            state = torch.cat((state, image_dist_tensor), 1)

        observation = torch.autograd.Variable(state)
        if opt.cuda:
            observation = observation.cuda()
        return observation

    def construct_entropy_state(self, model, index):
        current_sentence = loaders["train_loader"].dataset[index][0]
        current_sentence = Variable(torch.LongTensor(current_sentence))
        current_sentence.volatile = True

        if opt.cuda:
            current_sentence = current_sentence.cuda()
        preds = model(current_sentence)
        # preds = torch.sort(pred)

        # print(preds)
        return preds

    def construct_all_predictions(self, model):
        # all_predictions = torch.FloatTensor()
        all_predictions = None
        # if opt.cuda:
            # all_predictions = all_predictions.cuda()

        for i, train_data in enumerate(loaders["train_loader"]):
            sentences, targets = train_data
            features = Variable(sentences, requires_grad=False)

            if opt.cuda:
                features = features.cuda()

            preds = model(features)
            if i == 0:
                all_predictions = preds
            else:
                all_predictions = torch.cat((all_predictions, preds), dim=0)
        # print(all_predictions)
        data["all_predictions"] = all_predictions

    def feedback(self, action, model):
        reward = 0.
        is_terminal = False

        if action == 1:
            timer(self.query, (model,))
            new_performance = self.get_performance(model)
            reward = self.performance - new_performance

            if opt.reward_clip:
                reward = np.tanh(reward / 100)

            self.performance = new_performance
        else:
            reward = 0.

        # TODO fix this
        if self.queried_times >= self.budget or self.current_state >= len(self.order):
            # Return terminal
            return None, None, True

        print("> State {:2} Action {:2} - reward {:.4f} - accuracy {:.4f}".format(
            self.current_state, action, reward, self.performance))
        next_observation = timer(self.get_state, (model,))
        return reward, next_observation, is_terminal

    def query(self, model):
        self.construct_all_predictions(model)
        current = self.order[self.current_state]
        # construct_state = self.construct_entropy_state
        #
        # current_state = construct_state(model, current)
        # all_states = torch.cat([construct_state(model, index) for index in range(len(self.order))])
        current_state = data["all_predictions"][current].view(1, -1)
        all_states = data["all_predictions"]
        # print(current_state)
        # print(all_states)
        current_all_dist = pairwise_distances(current_state, all_states)
        similar_indices = torch.topk(current_all_dist, opt.selection_radius, 1, largest=False)[1]

        for index in similar_indices.data[0].cpu().numpy():
            image = loaders["train_loader"].dataset[index][0]
            caption = loaders["train_loader"].dataset[index][1]
            # There are 5 captions for every image
            loaders["active_loader"].dataset.add_single(image, caption)
            # Only count images as an actual request.
            # Reuslt is that we have 5 times as many training points as requests.
            self.queried_times += 1


        # for index in similar_indices[0]:
        #     image = loaders["train_loader"].dataset[5 * index][0]
        #     # There are 5 captions for every image
        #     for cap in range(5):
        #         caption = loaders["train_loader"].dataset[5 * index + cap][1]
        #         loaders["active_loader"].dataset.add_single(image, caption)
        #     # Only count images as an actual request.
        #     # Reuslt is that we have 5 times as many training points as requests.
        #     self.queried_times += 1

    def init_train_k_random(self, model, num_of_init_samples):
        for i in range(0, num_of_init_samples):
            current = self.order[(-1*(i + 1))]
            image = loaders["train_loader"].dataset[current][0]
            caption = loaders["train_loader"].dataset[current][1]
            loaders["active_loader"].dataset.add_single(image, caption)

        # TODO: delete used init samples (?)
        timer(model.train_model, (loaders["active_loader"], 30))

        print("Validation after training on random data: {}".format(model.validate(loaders["val_loader"])))

    def get_performance(self, model):
        # timer(self.train_model, (model, loaders["active_loader"]))
        timer(model.train_model, (loaders["active_loader"], opt.num_epochs))
        metrics = model.validate(loaders["val_loader"])
        performance = metrics["avg_loss"]

        if (self.queried_times % 20 == 0):
            if opt.embedding != 'static':
                self.encode_episode_data(model, loaders["train_loader"])
        return performance


    # def train(self, model, train_loader, epochs=opt.num_epochs):

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR
           decayed by 10 every 30 epochs"""
        lr = opt.learning_rate_vse * (0.1 ** (epoch // opt.lr_update))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
