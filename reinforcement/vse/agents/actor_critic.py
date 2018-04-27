import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from config import opt

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

GAMMA = 0.99

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(opt.topk + 1, 256)
        self.action_head = nn.Linear(256, 2)
        self.value_head = nn.Linear(256, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

class ActorCriticAgent:
    def __init__(self):
        self.policynetwork = Policy()
        if opt.cuda:
            self.policynetwork.cuda()

        self.optimizer = optim.Adam(self.policynetwork.parameters(), lr=3e-2)


    def get_action(self, state):
        probs, state_value = self.policynetwork(state)
        m = Categorical(probs)
        action = m.sample()
        self.policynetwork.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.data[0]

    def update(self, state, action, reward, next_state, terminal):
        self.policynetwork.rewards.append(reward)


    def finish_episode(self):
        R = 0
        saved_actions = self.policynetwork.saved_actions
        policy_losses = []
        value_losses = []
        rewards = []
        for r in self.policynetwork.rewards[::-1]:
            R = r + GAMMA * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for (log_prob, state_value), r in zip(saved_actions, rewards):
            reward = r - state_value
            policy_losses.append(-log_prob * reward)
            r = Variable(torch.Tensor([r]))
            if opt.cuda:
                r = r.cuda()
            value_losses.append(F.smooth_l1_loss(state_value, r))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()
        del self.policynetwork.rewards[:]
        del self.policynetwork.saved_actions[:]
