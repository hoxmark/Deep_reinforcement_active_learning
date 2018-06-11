import random

from agents import PolicyAgent, DQNTargetAgent, DQNAgent
from config import opt
import numpy as np
import torch
from torch import nn
import getpass
import sklearn
import datetime

from data.utils import visdom_logger, no_logger

# agent = DQNTargetAgent()

opt.state_size = 2
opt.hidden_size = 24
opt.cuda = False
opt.actions = 2
opt.batch_size_rl = 32
opt.gamma = 0
# opt.cuda = torch.cuda.is_available()
opt.cuda = False
opt.state_size = 2
opt.reward_clip = False
# agent = PolicyAgent()
agent = DQNAgent()
opt.logger_name = '{}_{}_test_{}'.format(getpass.getuser(), datetime.datetime.now().strftime("%d-%m-%y_%H:%M"), agent.__class__.__name__)

budget = 500
num_data = 10000
episodes = 10000
val_idx = 0.8

logger = visdom_logger()
# logger = no_logger()

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.fc1 = nn.Linear(opt.state_size, opt.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(opt.hidden_size, 1)


    def forward(self, inp):
        out = self.fc2(self.relu(self.fc1(inp)))
        return out

def generate_data():
    data = []
    rewards = []
    for i in range(num_data):
        class1 = random.random()
        class2 = 1 - class1
        probs = [class1, class2]
        data.append(torch.FloatTensor(np.sort(probs)).view(1, -1))
        reward = np.sum(-1 * (probs * np.log(probs)))

        # 0.69 is the max possible entropy, and should be the highest reward.
        reward = reward / 0.69314718056
        rewards.append(reward)

    train_x = data[0: int(num_data * 0.8)]
    train_y = rewards[0: int(num_data * 0.8)]

    val_x = data[int(num_data * 0.8):]
    val_y = rewards[int(num_data * 0.8):]
    return train_x, train_y, val_x, val_y
    # max = np.sum(sorted(rewards, reverse=True)[0:budget])

def batch(iterable, iterable2, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield (iterable[ndx:min(ndx + n, l)], iterable2[ndx:min(ndx + n, l)])

def train_regression():
    train_x, train_y, val_x, val_y = generate_data()
    model = LinearRegressionModel()
    features = torch.autograd.Variable(torch.cat(train_x))
    targets = torch.autograd.Variable(torch.FloatTensor(train_y))
    criterion = nn.MSELoss()
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(0, 100):
        for feature, target in batch(features, targets, 32):
            optimizer.zero_grad()
            output = model(feature)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        validate(model, epoch, val_x, val_y)


def validate(model, epoch, features, targets):
    features = torch.autograd.Variable(torch.cat(features))
    targets = torch.autograd.Variable(torch.FloatTensor(targets))
    corrects = 0
    threshold = 0.05
    for feature, target in batch(features, targets, 32):
        predicted = model(feature)
        # print(predicted)
        # print(target)
        diff = torch.abs(predicted.squeeze() - target)
        # print(diff < threshold)

        corrects += torch.sum(diff < threshold).data.long().tolist()[0]
    # print(corrects.data.cpu().numpy()[0])
    print("Epoch {} - {} / {} - {}% corrects".format(epoch, corrects, len(targets), (corrects/len(targets))*100))


def train_rl():
    train_x, train_y, val_x, val_y = generate_data()
    features = torch.autograd.Variable(torch.cat(train_x))
    targets = torch.FloatTensor(train_y)
    if opt.cuda:
        features, targets = features.cuda(), targets.cuda()
    running_mean_N = 5
    running_avg = []
    opt.max_ep_reward = max
    logger.parameters_summary()

    for episode in range(0, episodes):
        order = random.sample(list(range(0, int(num_data * val_idx))), int(num_data * val_idx))
        current_state = 0

        ep_reward = 0
        queried = 0

        while queried < budget:
            current_idx = order[current_state]
            state = features[current_idx].view(1, -1)

            reward = 0
            action = agent.get_action(state)
            if action == 1:
                reward = targets[current_idx] - 0.60
                ep_reward += reward
                queried += 1

            next_state = features[order[current_state + 1]].view(1, -1)
            if opt.cuda:
                next_state = next_state.cuda()
            # print("wqe", state, action, reward, next_state)

            # print(state, action, reward)
            agent.update(state, action, reward, next_state, False)

            state = next_state
            current_state += 1
        agent.finish_episode(episode)
        running_avg.append(ep_reward)
        if episode >= running_mean_N:
            running_avg.pop(0)
        print(running_avg)
        running_mean = np.mean(running_avg)

        metrics = {
            'running_avg': running_mean,
            'ep_reward': ep_reward
        }

        logger.dict_scalar_summary('test', metrics, episode)
        print("Episode {} - running reward {}".format(episode, running_mean))
# train_regression()
train_rl()
