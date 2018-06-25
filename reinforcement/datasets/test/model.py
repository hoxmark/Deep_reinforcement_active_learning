import torch
import copy
import numpy as np
from config import opt, data
from utils import entropy

class TestModel():
    def __init__(self):
        self.cumulative_reward = 0

    def reset(self):
        pass

    def forward(self, inp):
        pass

    def train_model(self, data, epochs):
        pass

    def validate(self, d):
        if hasattr(self, 'state_idx'):
            performance = data["train"][1][self.state_idx]
        else:
            performance = 0
        self.cumulative_reward += performance
        return {
            "performance": self.cumulative_reward
        }

    def performance_validate(self, d):
        """ Happens at the end of each episode. Reset the performance counter  """
        self.cumulative_reward = 0.
        return self.validate(d)

    def get_state(self, index):
        state = torch.Tensor(data["train"][0][index]).view(1, -1)
        self.state_idx = index
        if opt.cuda:
            state = state.cuda()
        return state

    def encode_episode_data(self):
        pass

    def query(self, index):
        return [index]

    def add_index(self, index):
        pass
