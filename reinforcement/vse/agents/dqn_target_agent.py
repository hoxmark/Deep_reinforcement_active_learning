import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from agents.prioritized_memory import Memory

from config import opt


# approximate Q function using Neural Network
# state is input and Q Value of each action is output of network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(opt.state_size, opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, opt.actions)
        )

    def forward(self, x):
        return self.fc(x)


# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and prioritized experience replay memory & target q network
class DQNTargetAgent():
    def __init__(self):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = opt.state_size
        self.action_size = opt.actions

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.memory_size = 20000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 5000
        self.train_start = 200
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = opt.batch_size_rl

        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        # create main model and target model
        self.model = DQN()
        self.model.apply(self.weights_init)
        self.target_model = DQN()

        if opt.cuda:
            self.model, self.target_model = self.model.cuda(), self.target_model.cuda()


        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

        # initialize target model
        self.update_target_model()

    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)

    # save sample (error,<s,a,r,s'>) to the replay memory
    def update(self, state, action, reward, next_state, done):
        target = self.model(state).data
        old_val = target[0][action]
        target_val = self.target_model(next_state).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))
        self.train_model()


    def train_model(self):
        if self.memory.tree.n_entries < self.train_start:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        minibatch, idxs, is_weights = self.memory.sample(self.batch_size)
        # print(minibatch)

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

        current_q_values = self.model(batch_state).gather(1, batch_action)
        max_next_q_values = self.target_model(batch_next_state).max(1)[0]
        expected_q_values = batch_reward + (self.discount_factor * max_next_q_values)
        # Undo volatility introduced above
        expected_q_values = Variable(expected_q_values.data).unsqueeze(1)

        if opt.cuda:
            expected_q_values = expected_q_values.cuda()

        loss = F.mse_loss(current_q_values, expected_q_values)
        errors = torch.abs(current_q_values - expected_q_values).data.cpu().numpy()
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def finish_episode(self, ep):
        self.update_target_model()
