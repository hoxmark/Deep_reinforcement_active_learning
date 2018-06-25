import numpy as np
import random
from collections import deque

from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import opt

# TODO put in params
# Hyper Parameters:
GAMMA = opt.gamma  # decay rate of past observations
OBSERVE = 0  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 10000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
# FINAL_EPSILON = 0
# INITIAL_EPSILON = 0.1
# or alternative:
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
EXPLORE = 100000.  # frames over which to anneal epsilon

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fcs = [nn.Linear(size, opt.hidden_size) for size in opt.data_sizes]
        self.out_fc = nn.Linear(opt.hidden_size, 2)
        self.activation = nn.ReLU()

        self.weights_init()
        
        if opt.cuda:
            self.fcs = [fc.cuda() for fc in self.fcs]
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
        forwards = self.activation(torch.add(*forwards))
        out = self.out_fc(forwards)
        return out

class DQNAgent:
    def __init__(self):
        self.replay_memory = deque()
        self.time_step = 0
        self.actions = opt.actions
        self.epsilon = INITIAL_EPSILON
        self.policynetwork = DQN()
        self.targetnetwork = DQN()
        self.optimizer = optim.Adam(self.policynetwork.parameters(), 0.01)

        # if opt.cuda:
            # self.policynetwork = self.policynetwork.cuda()
            # self.targetnetwork = self.targetnetwork.cuda()

        self.policynetwork.weights_init()
        self.update_target_model()

    def update_target_model(self):
        self.targetnetwork.load_state_dict(self.policynetwork.state_dict())

    def train_policynetwork(self):
        if len(self.replay_memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)

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
        max_next_q_values = self.targetnetwork(batch_next_state).max(1)[0]
        expected_q_values = batch_reward + (GAMMA * max_next_q_values)

        loss = F.mse_loss(current_q_values, expected_q_values.view(-1, 1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(self.policynetwork.fc1.weight.grad.nonzero())

        del batch_state, batch_action, batch_next_state, batch_reward, current_q_values, max_next_q_values, expected_q_values

    def update(self, current_state, action, reward, next_state, terminal):
        self.replay_memory.append(
            (current_state, action, reward, next_state, terminal))
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()
        global OBSERVE
        if self.time_step > OBSERVE:
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
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def finish_episode(self, episode):
        if episode % 10 == 0:
            self.update_target_model()
