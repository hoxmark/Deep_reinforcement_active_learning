import os
import random
from game import Game
from agents import DQNAgent, DQNTargetAgent, PolicyAgent, ActorCriticAgent, RandomAgent
from config import data, opt, loaders, global_logger
from models.vse import VSE
from data.utils import save_model, timer, load_external_model, average_vector, save_VSE_model,get_full_VSE_model

from models.simple_classifier import SimpleClassifier

from models.cnn import CNN
from models.svm import SVM
from sklearn import datasets, svm, metrics


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
    # classifier = VSE
    # classifier = CNN
    # classifier = SVM
    classifier = SimpleClassifier


    # if opt.embedding == 'static' and opt.dataset == 'vse':
    #     path_to_full_model ="{}/fullModel.pth.tar".format(opt.data_path)
    #     full_model = classifier()

    #     if os.path.isfile(path_to_full_model):
    #         get_full_VSE_model(full_model,path_to_full_model)
    #         game.encode_episode_data(full_model, loaders["train_loader"])
    #     else:
    #         print("No old model found, training a new one")
    #         print("Please wait... ")
    #         game.train_model(full_model, loaders["train_loader"], epochs=30)
    #         game.encode_episode_data(full_model, loaders["train_loader"])
    #         save_VSE_model(full_model.state_dict(), path=opt.data_path)

    for episode in range(start_episode, opt.episodes):
        model = classifier()
        game.reboot(model)
        print('##>>>>>>> Episode {} of {} <<<<<<<<<##'.format(episode, opt.episodes))
        terminal = False
        num_of_zero = 0

        state = game.get_state(model)
        while not terminal:
            action = agent.get_action(state)
            # action = random.randint(0,1)
            reward, next_state, terminal = game.feedback(action, model)
            if terminal:
                agent.finish_episode(episode)
                break
            agent.update(state, action, reward, next_state, terminal)
            # print("\n")
            if (action == 1):
                print(state)
                lg.scalar_summary("last_episode_performance", game.performance, game.queried_times)
                # print(state)
                # Reset the model every time we add to train set
                model = classifier()   #SHould this be done? # Yes I think so
            else:
                num_of_zero += 1
            state = next_state



        # Reset model
        model = classifier()
        # print("len:")
        # print(len(loaders["active_loader"].dataset))
        print(timer(model.train_model, (loaders["active_loader"], 100)))

        # model.train_model(loaders["active_loader"], 100)
        metrics = timer(model.performance_validate, (loaders["val_loader"],))
        # metrics = model.performance_validate(loaders["val_loader"])
        # print(len(loaders["val_loader"].dataset))

        lg.dict_scalar_summary('episode-validation', metrics, episode)
        # lg.scalar_summary('episode-validation', metrics, episode)
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
