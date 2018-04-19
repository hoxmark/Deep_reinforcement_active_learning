import torch.nn as nn
import torch.nn.functional as F

from config import params

class DQN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.IN_SIZE = 2
        self.HIDDEN_SIZE = 256
        self.OUT_SIZE = params["ACTIONS"]

        self.fc = nn.Linear(self.IN_SIZE, self.HIDDEN_SIZE)
        self.fc2 = nn.Linear(self.HIDDEN_SIZE, self.OUT_SIZE)
        self.relu = nn.ReLU()


    def forward(self, inp):
        x = F.relu(self.fc(inp))
        x = self.fc2(x)
        return x
