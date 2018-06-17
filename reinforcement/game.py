import numpy as np
import random
import torch
import copy
from pprint import pprint
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
from config import opt, data
from utils import timer


class Game:
    def reboot(self, model):
        """resets the Game Object, to make it ready for the next episode """
        data["active"] = tuple(([] for i in range(len(data["train"]))))

        self.order = random.sample(list(range(0, opt.data_len)), opt.data_len)
        self.budget = opt.budget
        self.queried_times = 0
        self.current_state = 0
        data["train_deleted"] = copy.deepcopy(data["train"])
        self.init_train_k_random(model, opt.init_samples)
        timer(model.encode_episode_data, ())
        metrics = model.validate(data["dev"])
        self.performance = metrics["performance"]

    def init_train_k_random(self, model, num_samples):
        for i in range(0, num_samples):
            current = self.order[(-1*(i + 1))]
            model.add_index(current)
        # TODO: delete used init samples (?)
        timer(model.train_model, (data["active"], opt.num_epochs))

    def get_state(self, model):
        with torch.no_grad():
            current_idx = self.order[self.current_state]
            state = model.get_state(current_idx)
            state = Variable(state).view(1, -1)
            state = state.sort(descending=True)[0]
            if opt.cuda:
                state = state.cuda()

            self.current_state += 1
            return state

    def feedback(self, action, model):
        reward = 0.
        is_terminal = False
        if action == 1:
            timer(self.query, (model,))
            new_performance = self.get_performance(model)
            reward = new_performance - self.performance - opt.reward_threshold
            # if opt.reward_clip:
                # reward = np.tanh(reward / 100)
            self.performance = new_performance
        else:
            reward = 0.

        if self.queried_times >= self.budget or self.current_state >= len(self.order):
            return reward, None, True

        next_observation = timer(self.get_state, (model,))


        return reward, next_observation, is_terminal

    def query(self, model):
        current = self.order[self.current_state]
        similar_indices = model.query(current)
        new_data = [*data["train_deleted"]]
        for id in reversed(sorted(similar_indices)):
            for i, d in enumerate(new_data):
                new_data[i] = np.delete(new_data[i], id, axis=0)
            self.order.remove(id)
        data["train_deleted"] = new_data

        self.queried_times += opt.selection_radius
        self.order = list(map(lambda x: x - np.where(np.array([x]) > similar_indices)[0].shape[0], self.order))

    def get_performance(self, model):
        timer(model.train_model, (data["active"], opt.num_epochs))
        metrics = timer(model.validate, (data["dev"],))
        performance = metrics["performance"]
        timer(model.encode_episode_data, ())
        return performance
