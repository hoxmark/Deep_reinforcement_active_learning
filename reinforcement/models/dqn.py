import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, params):
        super(DQN, self).__init__()
        self.IN_SIZE = 1293
        self.HIDDEN_SIZE = 400
        self.OUT_SIZE = params["ACTIONS"]

        self.fc = nn.Linear(self.IN_SIZE, self.HIDDEN_SIZE)
        self.fc2 = nn.Linear(self.HIDDEN_SIZE, self.OUT_SIZE)


    def forward(self, inp):
        x = nn.ReLU(self.fc(inp))
        x = self.fc2(inp)
        return x
