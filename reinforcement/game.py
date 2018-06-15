import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

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
        self.init_train_k_random(model, opt.init_samples)
        model.encode_episode_data()
        # self.avg_entropy_in_train_loader = self.get_avg_entropy_in_train_loader(loaders["train_loader"])
        metrics = model.validate(data["dev"])
        self.performance = metrics["performance"]

    def init_train_k_random(self, model, num_samples):
        for i in range(0, num_samples):
            current = self.order[(-1*(i + 1))]
            model.add_index(current)
        # TODO: delete used init samples (?)
        timer(model.train_model, (data["active"], opt.num_epochs))


    def get_state(self, model):
        current_idx = self.order[self.current_state]
        state = model.get_state(current_idx)
        # state = Variable(torch.FloatTensor(state).view(1, -1))
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

        print("> State {:2} Action {:2} - reward {:.4f} - performance {:.4f}".format(
            self.current_state, action, reward, self.performance))
        next_observation = timer(self.get_state, (model,))
        if self.queried_times >= self.budget or self.current_state >= len(self.order):
            is_terminal = True

        return reward, next_observation, is_terminal

    def query(self, model):
        current = self.order[self.current_state]
        model.query(current)
        self.queried_times += opt.selection_radius

    def get_performance(self, model):
        timer(model.train_model, (data["active"], opt.num_epochs))
        metrics = timer(model.validate, (data["dev"],))
        performance = metrics["performance"]
        model.encode_episode_data()
        return performance
