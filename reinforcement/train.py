from agent import RobotCNNDQN
from models.cnn import CNN
from game import Game
import utils

def train(data, params):
    if params["EMBEDDING"] == "static":
        w2v = utils.load_word2vec(data)
        data["w2v"] = w2v
    agent = RobotCNNDQN(params)
    model = CNN(data, params)
    model.init_model()
    game = Game(data, params)

    for episode in range(params["EPISODES"]):
        terminal = False
        game.reboot()
        model.init_model()
        print('>>>>>>> Current game round ', episode, 'Maximum ', params["EPISODES"])

        while not terminal:
            observation = game.get_frame(model)
            action = agent.get_action(observation)
            print('> Action', action)
            reward, observation2, terminal = game.feedback(action, model)
            if terminal:
                break
            # print('> Reward', reward)

            agent.update(observation, action, reward, observation2, terminal)


    # Test agent here
