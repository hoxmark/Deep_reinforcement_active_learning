import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from config import opt


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.repr_fc = nn.Linear(opt.data_size, opt.hidden_size)
        self.pred_fc = nn.Linear(opt.pred_size, opt.hidden_size)
        self.out_fc = nn.Linear(opt.hidden_size, 2)
        self.activation = nn.ReLU()
        self.weights_init()

        self.saved_log_probs = []
        self.rewards = []

    def weights_init(self):
        torch.nn.init.xavier_normal_(self.repr_fc.weight)
        torch.nn.init.xavier_normal_(self.pred_fc.weight)
        torch.nn.init.xavier_normal_(self.out_fc.weight)

    def forward(self, x):
        img = inp.narrow(1, 0, opt.data_size)
        pred = inp.narrow(1, opt.data_size, opt.pred_size)
        img_f = self.repr_fc(img)
        pred_f = self.pred_fc(pred)
        h = img_f + pred_f
        out = self.activation(h)
        out = self.out_fc(out)
        return F.softmax(out, dim=1)

class PolicyAgent:
    def __init__(self):
        self.policynetwork = Policy()
        self.optimizer = optim.Adam(self.policynetwork.parameters(), lr=1e-2)
        self.running_reward = 10
        self.gamma = opt.gamma

        if opt.cuda:
            self.policynetwork.cuda()

    def get_action(self, state):
        probs = self.policynetwork(state)
        m = Categorical(probs)
        action = m.sample()
        self.policynetwork.saved_log_probs.append(m.log_prob(action))
        del state
        action = action.item()
        return action

    def update(self, state, action, reward, next_state, terminal):
        self.policynetwork.rewards.append(reward)

    def finish_episode(self, episode):
        # TODO time this
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policynetwork.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        if opt.cuda:
            rewards = rewards.cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps.item())
        # print(rewards)
        if opt.cuda:
            rewards = rewards.cuda()
        for log_prob, reward in zip(self.policynetwork.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policynetwork.rewards[:]
        del self.policynetwork.saved_log_probs[:]
