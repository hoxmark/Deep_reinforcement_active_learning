import numpy as np
import random
from collections import deque

from torch import optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from config import opt

# TODO put in params
# Hyper Parameters:
GAMMA = 0  # decay rate of past observations
OBSERVE = 32  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 10000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
# FINAL_EPSILON = 0
# INITIAL_EPSILON = 0.1
# or alternative:
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
UPDATE_TIME = 100
EXPLORE = 100000.  # frames over which to anneal epsilon

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.IN_SIZE = opt.state_size
        self.HIDDEN_SIZE = opt.hidden_size
        self.OUT_SIZE = opt.actions

        self.fc1 = nn.Linear(self.IN_SIZE, self.HIDDEN_SIZE)
        self.fc2 = nn.Linear(self.HIDDEN_SIZE, self.OUT_SIZE)
        self.activation = nn.Tanh() if opt.reward_clip else nn.ReLU()

        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc1.weight.data.normal_(0, 0.1)   # initialization


    def forward(self, inp):
        x = self.activation(self.fc1(inp))
        x = self.fc2(x)
        return x



class DQNAgent:
    def __init__(self):
        self.replay_memory = deque()
        self.time_step = 0
        self.actions = opt.actions
        self.epsilon = INITIAL_EPSILON
        self.policynetwork = DQN()
        self.optimizer = optim.Adam(self.policynetwork.parameters(), 0.01)

        if opt.cuda:
            self.policynetwork = self.policynetwork.cuda()

    def initialise(self):
        self.policynetwork = DQN()

    def load_policynetwork(self, old_modal):
        self.policynetwork.load_state_dict(old_modal)

    def train_policynetwork(self):
        if len(self.replay_memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)

        batch_state, batch_action, batch_reward, batch_next_state, batch_terminal = zip(*minibatch)
        batch_state = torch.cat(batch_state)
        batch_next_state = torch.cat(batch_next_state)

        batch_action = Variable(torch.LongTensor(list(batch_action)).unsqueeze(1))
        batch_reward = Variable(torch.FloatTensor(list(batch_reward)))

        if opt.cuda:
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        current_q_values = self.policynetwork(batch_state).gather(1, batch_action)
        with torch.no_grad():
            max_next_q_values = self.policynetwork(batch_next_state).max(1)[0]

        expected_q_values = batch_reward + (GAMMA * max_next_q_values)

        # if opt.cuda:
            # expected_q_values = expected_q_values.cuda()
        loss = F.mse_loss(current_q_values, expected_q_values.view(-1, 1))


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        with torch.no_grad():
            action = 0
            if random.random() <= self.epsilon:
                action = random.randrange(self.actions)
            else:
                qvalue = self.policynetwork(state)
                action = np.argmax(qvalue.data[0])
            # change epsilon
            if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
                self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            return action

    def finish_episode(self, episode):
        pass
