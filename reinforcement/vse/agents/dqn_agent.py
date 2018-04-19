import tensorflow as tf
import numpy as np
import random
from collections import deque
from models.dqn import DQN
from torch import optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from config import opt

# TODO put in params
# Hyper Parameters:
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 32  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 1000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
# FINAL_EPSILON = 0
# INITIAL_EPSILON = 0.1
# or alternative:
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
UPDATE_TIME = 100
EXPLORE = 100000.  # frames over which to anneal epsilon


class DQNAgent:
    def __init__(self):
        self.replay_memory = deque()
        self.time_step = 0
        self.actions = opt.actions
        self.epsilon = INITIAL_EPSILON
        self.policynetwork = DQN()

        if opt.cuda:
            self.policynetwork = self.policynetwork.cuda()

    def initialise(self):
        self.policynetwork = DQN()

    def train_policynetwork(self):
        optimizer = optim.Adam(self.policynetwork.parameters(), 0.001)
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)

        batch_state, batch_action, batch_reward, batch_next_state, batch_terminal = zip(*minibatch)
        batch_state = torch.cat(batch_state)
        batch_next_state = torch.cat(batch_next_state)
        batch_next_state.volatile = True

        batch_action = Variable(torch.LongTensor(list(batch_action)).unsqueeze(1))
        batch_reward = Variable(torch.FloatTensor(list(batch_reward)))

        if opt.cuda:
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        current_q_values = self.policynetwork(batch_state).gather(1, batch_action)
        max_next_q_values = self.policynetwork(batch_next_state).max(1)[0]
        expected_q_values = batch_reward + (GAMMA * max_next_q_values)
        # Undo volatility introduced above
        expected_q_values = Variable(expected_q_values.data)

        if opt.cuda:
            expected_q_values = expected_q_values.cuda()
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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
            action = np.argmax(qvalue.data[0])
        # change epsilon
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def finish_episode(self):
        pass
