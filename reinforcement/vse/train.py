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
            if (action == 1):                
                lg.scalar_summary("performance_in_episode_{}".format(episode), game.performance, game.queried_times)
            lg.scalar_summary("action_choice_in_episode_{}".format(episode), action, game.current_state)

        episode_validation = game.performace_validate(model)
        lg.scalar_summary("episode-validation", episode_validation, episode)
        
        lg.scalar_summary("episode-acc", game.performance, episode)
