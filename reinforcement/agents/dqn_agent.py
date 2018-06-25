import numpy as np
import random
from collections import deque

from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import opt

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fcs = nn.ModuleList([nn.Linear(size, opt.hidden_size) for size in opt.data_sizes])
        self.out_fc = nn.Linear(opt.hidden_size, 2)
        self.activation = nn.ReLU()

        self.weights_init()

        if opt.cuda:
            self.cuda()

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
        return out

class DQNAgent:
    def __init__(self):
        self.replay_memory = deque()
        self.time_step = 0
        self.observe = 32
        self.replay_memory_size = 10000
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.explore = 100000.

        self.actions = opt.actions
        self.epsilon = INITIAL_EPSILON
        self.policynetwork = DQN()
        self.targetnetwork = DQN()

        self.update_target_network()

        self.optimizer = optim.RMSprop(self.policynetwork.parameters())


    def update_target_network(self):
        self.targetnetwork.load_state_dict(self.policynetwork.state_dict())

    def train_policynetwork(self):
        if len(self.replay_memory) < opt.batch_size_rl:
            return
        minibatch = random.sample(self.replay_memory, opt.batch_size_rl)

        batch_state, batch_action, batch_reward, batch_next_state, batch_terminal = zip(*minibatch)
        batch_state = torch.cat(batch_state)
        batch_next_state = torch.cat(batch_next_state)

        batch_action = torch.LongTensor(list(batch_action)).unsqueeze(1)
        batch_reward = torch.FloatTensor(list(batch_reward))

        if opt.cuda:
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        current_q_values = self.policynetwork(batch_state).gather(1, batch_action)
        max_next_q_values = self.targetnetwork(batch_next_state).max(1)[0].detach()
        expected_q_values = batch_reward + (opt.gamma * max_next_q_values)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.view(-1, 1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del batch_state, batch_action, batch_next_state, batch_reward, current_q_values, max_next_q_values, expected_q_values

    def update(self, current_state, action, reward, next_state, terminal):
        self.replay_memory.append(
            (current_state, action, reward, next_state, terminal))
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.popleft()
        if self.time_step > self.observe:
            self.train_policynetwork()

        self.time_step += 1

    def get_action(self, state):
        action = 0
        if random.random() <= self.epsilon:
            action = random.randrange(self.actions)
        else:
            qvalue = self.policynetwork(state)
            action = np.argmax(qvalue.data[0]).item()
        # change epsilon
        if self.epsilon > self.final_epsilon and self.time_step > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        return action

    def finish_episode(self, episode):
        if episode % 10 == 0:
            self.update_target_network()
