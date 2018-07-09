import os
import random
from game import Game
from agents import DQNAgent, DQNTargetAgent, PolicyAgent, ActorCriticAgent, RandomAgent
from config import data, opt, loaders, global_logger
from utils import save_model, timer, load_external_model, average_vector, save_VSE_model,get_full_VSE_model

def train(classifier):
    lg = global_logger["lg"]

    if opt.agent == 'policy':
        agent = PolicyAgent()
    elif opt.agent == 'dqn':
        agent = DQNAgent()
    elif opt.agent == 'dqn_target':
        agent = DQNTargetAgent()
    elif opt.agent == 'actor_critic':
        agent = ActorCriticAgent()
    elif opt.agent == 'random':
        agent = RandomAgent()
    else:
        agent = DQNAgent()

    start_episode = 0

    # load old model
    file_name = opt.load_model_name
    if file_name != "":
        old_model = load_external_model(file_name)
        start_episode = int(file_name.split('/')[1])
        agent.load_policynetwork(old_model)

    game = Game()
    model = classifier()
    for episode in range(start_episode, opt.episodes):
        model.reset()
        game.reboot(model)
        print('##>>>>>>> Episode {} of {} <<<<<<<<<##'.format(episode, opt.episodes))
        terminal = False
        num_of_zero = 0

        state = game.get_state(model)
        first_log = True
        cum_reward = 0
        while not terminal:
            action = agent.get_action(state)
            reward, next_state, terminal = game.feedback(action, model)
            if not terminal:
                agent.update(state, action, reward, next_state, terminal)

            cum_reward += reward
            if (action == 1):
                print("> State {:2} Action {:2} - reward {:.4f} - performance {:.4f}".format(game.current_state, action, reward, game.performance))
                # print(state)
                step = 0 if first_log else game.queried_times
                timer(lg.scalar_summary, ("last_episode_performance", game.performance, step))
                first_log = False
            else:
                num_of_zero += 1

            del state
            state = next_state
            if terminal:
                agent.finish_episode(episode)
                break

        # Reset model
        model.reset()
        timer(model.train_model, (data["active"], opt.full_epochs))
        metrics = timer(model.performance_validate, (data["dev"],))

        lg.dict_scalar_summary('episode-validation', metrics, episode)
        lg.scalar_summary('episode-cum-reward', cum_reward, episode)
        lg.scalar_summary('performance', game.performance, episode)
        lg.scalar_summary('number-of-0-actions', num_of_zero, episode)

        # save_VSE_model(model.state_dict(), path=opt.data_path)
        # new_m = VSE()
        # path_to_full_model ="{}/fullModel.pth.tar".format(opt.data_path)
        # get_full_VSE_model(new_m,path_to_full_model)

        # if opt.load_model_name != "":
        #     old_model = load_external_model("Episode_0_performance_12.24")
        #     agent.set_policynetwork(old_model)

        # Save the model TODO
        # if opt.agent != 'random':
        #     model_path = '{}/{}'.format(opt.agent, str(episode).zfill(4) )
        #     model_name = '{:.2f}'.format(metrics["performance"])
        #     path = "{}/{}".format(model_path, model_name)
        #     print(path)
        #     save_model(path, agent.policynetwork.cpu())

        #     # Move it back to the GPU.
        #     if opt.cuda:
        #         agent.policynetwork.cuda()
