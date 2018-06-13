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

num_classes = 20
num_data = 10000
val_idx = 0.8
budget = 500
episodes = 10000

opt.state_size = num_classes
opt.hidden_size = 320
opt.cuda = False
opt.actions = 2
opt.batch_size_rl = 32
opt.gamma = 0.9
# opt.cuda = torch.cuda.is_available()
opt.cuda = False
opt.reward_clip = False

# agent = PolicyAgent()
agent = DQNAgent()
opt.logger_name = '{}_{}_test_{}'.format(getpass.getuser(), datetime.datetime.now().strftime("%d-%m-%y_%H:%M"), agent.__class__.__name__)

# logger = visdom_logger()
logger = no_logger()

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.fc1 = nn.Linear(opt.state_size, opt.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(opt.hidden_size, 1)


    def forward(self, inp):
        out = self.fc2(self.relu(self.fc1(inp)))
        return out



def generate_data(n_classes, num_data, val_idx):
    softmax_scale_factor = 10
    data = []
    rewards = []
    max_reward = torch.Tensor([1/n_classes for i in range(0, n_classes)])
    max_reward = torch.mul(max_reward, torch.log(max_reward))
    max_reward = torch.sum(max_reward)
    max_reward = max_reward * -1
    print(max_reward)
    for i in range(num_data):
        r = random.random()
        probs = [random.random() * softmax_scale_factor for i in range(n_classes)]
        probs = torch.Tensor(probs)
        probs = torch.nn.functional.softmax(probs, dim=0)
        probs = probs.sort()[0]
        data.append(probs.view(1, -1))

        # Calculate entropy
        reward = torch.mul(probs, torch.log(probs))
        reward = torch.sum(reward)
        reward = reward * -1

        # Scale with max reward to get it in range [0, 1]
        reward = reward / max_reward
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
    num_classes = 2
    num_data = 10000
    val_idx = 0.8

    train_x, train_y, val_x, val_y = generate_data(num_classes, num_data, val_idx)
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
    train_x, train_y, val_x, val_y = generate_data(num_classes, num_data, val_idx)
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
        # print(running_avg)
        running_mean = np.mean(running_avg)

        metrics = {
            'running_avg': running_mean,
            'ep_reward': ep_reward.item()
        }

        logger.dict_scalar_summary('test', metrics, episode)
        print("Episode {} - running reward {}".format(episode, running_mean))
# train_regression()
train_rl()
