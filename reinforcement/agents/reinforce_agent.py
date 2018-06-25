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
        self.fcs = nn.ModuleList([nn.Linear(size, opt.hidden_size) for size in opt.data_sizes])
        self.out_fc = nn.Linear(opt.hidden_size, 2)
        self.activation = nn.ReLU()

        self.weights_init()

        if opt.cuda:
            self.cuda()

        self.saved_log_probs = []
        self.rewards = []

    def weights_init(self):
        for layer in self.fcs:
            torch.nn.init.xavier_normal_(layer.weight)
        torch.nn.init.xavier_normal_(self.out_fc.weight)

    def forward(self, inp):
        if opt.cuda:
            inp = inp.cuda()
        inps = [inp.narrow(1, int(start), int(stop)) for (start, stop) in zip(np.cumsum(opt.data_sizes) - opt.data_sizes, opt.data_sizes)]
        forwards = [fc(d) for fc, d in zip(self.fcs, inps)]
        if len(opt.data_sizes) > 1:
            forwards = self.activation(torch.add(*forwards))
        else:
            forwards = self.activation(forwards[0])
        out = self.out_fc(forwards)
        return F.softmax(out, dim=1)

class PolicyAgent:
    def __init__(self):
        self.policynetwork = Policy()
        self.optimizer = optim.Adam(self.policynetwork.parameters(), opt.learning_rate_rl)
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
