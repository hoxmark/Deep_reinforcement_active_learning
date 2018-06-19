import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from config import opt


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(opt.state_size, opt.hidden_size)
        self.affine2 = nn.Linear(opt.hidden_size, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class PolicyAgent:
    def __init__(self):
        self.policynetwork = Policy()
        self.optimizer = optim.Adam(self.policynetwork.parameters(), lr=1e-2)
        self.running_reward = 10
        self.gamma = opt.gamma

        if opt.cuda:
            self.policynetwork.cuda()

    def get_action(self, state):
        state = Variable(state.data)
        probs = self.policynetwork(state)
        m = Categorical(probs)
        action = m.sample()
        self.policynetwork.saved_log_probs.append(m.log_prob(action))
        del state
        action = action.data[0]
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
