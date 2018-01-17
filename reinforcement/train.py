from agent import RobotCNNDQN
from models.cnn import CNN
from game import Game
import utils

def train(data, params):
    # w2v = utils.load_word2vec(data)
    # data["w2v"] = w2v
    agent = RobotCNNDQN(params)
    model = CNN(data, params)

    game = Game(data, params)




    for episode in range(params["EPISODES"]):
        terminal = False
        game.reboot()

        while not terminal:
            print('>>>>>>> Current game round ', episode, 'Maximum ', params["EPISODES"])
            observation = game.get_frame(model)
            action = agent.get_action(observation)
            print('> Action', action)
            reward, observation2, terminal = game.feedback(action, model)
            print('> Reward', reward)

            agent.update(observation, action, reward, observation2, terminal)


    # Test agent here
