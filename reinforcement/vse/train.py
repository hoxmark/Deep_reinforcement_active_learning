from agent import RobotCNNDQN
from models.vse import VSE
from game import Game
import utils
from config import data, opt, loaders, global_logger
from evaluation import encode_data

def train():
    model = VSE()

    lg = global_logger["lg"]

    img_embs, cap_embs = encode_data(model, loaders["train_loader"])
    img_embs_val, cap_embs_val = encode_data(model, loaders["val_loader"])

    agent = RobotCNNDQN()

    game = Game()

    for episode in range(opt.episodes):
        terminal = False
        num_of_states = 0
        game.reboot(model)
        # TODO init model
        # model.init_model()
        print('>>>>>>> EPISODE', episode, 'Of ', opt.episodes)
        observation = game.get_state(model)
        while not terminal:
            action = agent.get_action(observation)
            reward, observation2, terminal = game.feedback(action, model)
            if terminal:
                break

            agent.update(observation, action, reward, observation2, terminal)
            print("\n")
            observation = observation2
            lg.scalar_summary("performance_in_episode_{}".format(episode), game.performance, num_of_states)
            num_of_states +=1
            
        lg.scalar_summary("episode-acc", game.performance, episode)
