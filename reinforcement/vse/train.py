from game import Game
from agents import DQNAgent, PolicyAgent
from config import data, opt, loaders, global_logger
from models.vse import VSE
from data.evaluation import encode_data
from data.utils import save_model, timer


def train():
    lg = global_logger["lg"]


    if opt.agent == 'policy':
        agent = PolicyAgent()
    elif opt.agent == 'dqn':
        agent = DQNAgent()
    else:
        agent = DQNAgent()

    game = Game()

    for episode in range(opt.episodes):
        model = VSE()
        game.reboot(model)
        print('##>>>>>>> Episode {} of {} <<<<<<<<<##'.format(episode, opt.episodes))
        terminal = False


        state = game.get_state(model)
        while not terminal:
            action = agent.get_action(state)
            reward, next_state, terminal = game.feedback(action, model)
            if terminal:
                break

            agent.update(state, action, reward, next_state, terminal)
            print("\n")
            state = next_state
            if (action == 1):
                lg.scalar_summary("performance_in_episode/{}".format(episode), game.performance, game.queried_times)

        agent.finish_episode()

        # Logging each episode:
        (performance, r1, r5, r10, r1i, r5i, r10i) = utils.timer(game.performance_validate, (model,))
        lg.scalar_summary("episode-validation/sum", performance, episode)
        lg.scalar_summary("episode-validation/r1", r1, episode)
        lg.scalar_summary("episode-validation/r5", r5, episode)
        lg.scalar_summary("episode-validation/r10", r10, episode)
        lg.scalar_summary("episode-validation/r1i", r1i, episode)
        lg.scalar_summary("episode-validation/r5i", r5i, episode)
        lg.scalar_summary("episode-validation/r10i", r10i, episode)
        lg.scalar_summary("episode-loss", game.performance, episode)

        # Save the model
        model_name = 'Episode_{}_performance_{:.2f}'.format(episode, performance)
        save_model(model_name, agent.policynetwork.cpu())

        # Move it back to the GPU.
        if opt.cuda:
            agent.policynetwork.cuda()
