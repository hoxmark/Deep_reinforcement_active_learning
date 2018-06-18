import copy
from config import opt, data

class TestModel():
    def reset(self):
        pass

    def forward(self, inp):
        pass

    def train_model(self, data, epochs):
        pass

    def validate(self, data):
        return {
            "performance": 0
        }

    def performance_validate(self, data):
        return self.validate(data)

    def get_state(self, index):
        return data["train"][0][index]

    def encode_episode_data(self):
        pass

    def query(self, index):
        return [index]

    def add_index(self, index):
        pass
