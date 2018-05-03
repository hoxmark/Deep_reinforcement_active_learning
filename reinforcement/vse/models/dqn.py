import torch.nn as nn
import torch.nn.functional as F

from config import opt

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.IN_SIZE = opt.state_size
        self.HIDDEN_SIZE = opt.hidden_size
        self.OUT_SIZE = opt.actions

        self.fc1 = nn.Linear(self.IN_SIZE, self.HIDDEN_SIZE)
        self.fc2 = nn.Linear(self.HIDDEN_SIZE, self.OUT_SIZE)
        self.relu = nn.ReLU()

        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc1.weight.data.normal_(0, 0.1)   # initialization


    def forward(self, inp):
        x = F.relu(self.fc1(inp))
        x = self.fc2(x)
        return x
