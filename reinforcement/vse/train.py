import os
from game import Game
from agents import DQNAgent, DQNTargetAgent, PolicyAgent, ActorCriticAgent, RandomAgent
from config import data, opt, loaders, global_logger
from models.vse import VSE
from data.evaluation import encode_data
from data.utils import save_model, timer, load_external_model, average_vector, save_VSE_model,get_full_VSE_model


def train():
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

    if opt.embedding == 'static' :
        path_to_full_model ="{}/fullModel.pth.tar".format(opt.data_path)
        full_model = VSE()

        if os.path.isfile(path_to_full_model):
            get_full_VSE_model(full_model,path_to_full_model)
            game.encode_episode_data(full_model, loaders["train_loader"])
        else:
            print("No old model found, training a new one")
            print("Please wait... ")
            game.train_model(full_model, loaders["train_loader"], epochs=30)
            game.encode_episode_data(full_model, loaders["train_loader"])
            save_VSE_model(full_model.state_dict(), path=opt.data_path)

    for episode in range(start_episode, opt.episodes):
        model = VSE()
        game.reboot(model)
        print('##>>>>>>> Episode {} of {} <<<<<<<<<##'.format(episode, opt.episodes))
        terminal = False

        state = game.get_state(model)
        while not terminal:
            action = agent.get_action(state)
            reward, next_state, terminal = game.feedback(action, model)
            if terminal:
                agent.finish_episode(episode)
                break
                
            agent.update(state, action, reward, next_state, terminal)
            print("\n")
            state = next_state
            if (action == 1):
                lg.scalar_summary("last_episode_performance", game.performance, game.queried_times)
                # Reset the model every time we add to train set
                model = VSE()

        # Reset model
        model = VSE()
        game.train_model(model, loaders["active_loader"], epochs=30)

        (performance, r1, r5, r10, r1i, r5i, r10i) = timer(game.performance_validate, (model,))
        lg.scalar_summary("episode-validation/sum", performance, episode)
        lg.scalar_summary("episode-validation/r1", r1, episode)
        lg.scalar_summary("episode-validation/r5", r5, episode)
        lg.scalar_summary("episode-validation/r10", r10, episode)
        lg.scalar_summary("episode-validation/r1i", r1i, episode)
        lg.scalar_summary("episode-validation/r5i", r5i, episode)
        lg.scalar_summary("episode-validation/r10i", r10i, episode)
        lg.scalar_summary("episode-validation/loss", game.performance, episode)

        # print("first:")
        # print(model.state_dict())
        # save_VSE_model(model.state_dict(), path=opt.data_path)

        # new_m = VSE()
        # path_to_full_model ="{}/fullModel.pth.tar".format(opt.data_path)

        # get_full_VSE_model(new_m,path_to_full_model)

        # print("last:")
        # print(new_m.state_dict())

        # quit()

        # print(agent.policynetwork)
        # if opt.load_model_name != "":
        #     old_model = load_external_model("Episode_0_performance_12.24")
        #     print(old_model.cpu())
        #     agent.set_policynetwork(old_model)

        # print("ETter")
        # print(agent.policynetwork)
        # Save the model
        if opt.agent != 'random':
            model_path = '{}/{}'.format(opt.agent, str(episode).zfill(4) )
            model_name = '{:.2f}'.format(performance)
            path = "{}/{}".format(model_path, model_name)
            print(path)
            save_model(path, agent.policynetwork.cpu())

            # Move it back to the GPU.
            if opt.cuda:
                agent.policynetwork.cuda()
