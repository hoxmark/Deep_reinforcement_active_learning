import random

class RandomAgent:
    def __init__(self):
        self.actions = 2

    def update(self, current_state, action, reward, next_state, terminal):
        pass

    def get_action(self, state):
        return random.randrange(self.actions)

    def finish_episode(self, episode):
        pass
