from agent import RobotCNNDQN
from models.vse import VSE
from game import Game
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
                lg.scalar_summary("performance_in_episode/{}".format(episode), game.performance, game.queried_times)

        # Logging each episode: 
        (performance, r1, r5, r10, r1i, r5i, r10i) = timer(game.performance_validate, (model,))
        lg.scalar_summary("episode-validation/sum", performance, episode)
        lg.scalar_summary("episode-validation/r1", r1, episode)
        lg.scalar_summary("episode-validation/r5", r5, episode)
        lg.scalar_summary("episode-validation/r10", r10, episode)
        lg.scalar_summary("episode-validation/r1i", r1i, episode)
        lg.scalar_summary("episode-validation/r5i", r5i, episode)
        lg.scalar_summary("episode-validation/r10i", r10i, episode)        
        lg.scalar_summary("episode-loss", game.performance, episode)
