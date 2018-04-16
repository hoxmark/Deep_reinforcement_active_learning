from agent import RobotCNNDQN
from models.vse import VSE
from game import Game
import utils
from config import data, opt, loaders, global_logger
from evaluation import encode_data

def train():
    lg = global_logger["lg"]
    agent = RobotCNNDQN()
    game = Game()

    for episode in range(opt.episodes):
        model = VSE()
        game.reboot(model)

        print('>>>>>>> Episode', episode, 'Of ', opt.episodes)
        terminal = False
        observation = game.get_state(model)
        while not terminal:
            action = agent.get_action(observation)
            reward, observation2, terminal = game.feedback(action, model)
            if terminal:
                break

            agent.update(observation, action, reward, observation2, terminal)
            print("\n")
            observation = observation2
            lg.scalar_summary("performance_in_episode_{}".format(episode), game.performance, game.current_state)

        lg.scalar_summary("episode-acc", game.performance, episode)
