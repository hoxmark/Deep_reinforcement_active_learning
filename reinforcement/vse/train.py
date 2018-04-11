from agent import RobotCNNDQN
from models.cnn import CNN
from game import Game
import utils
from config import data, params

def train():
    if params["EMBEDDING"] == "static":
        utils.load_word2vec()

    lg = utils.init_logger()
    agent = RobotCNNDQN()
    model = CNN()
    game = Game()

    for episode in range(params["EPISODES"]):
        terminal = False
        game.reboot()
        model.init_model()
        print('>>>>>>> Current game round ', episode, 'Maximum ', params["EPISODES"])

        while not terminal:
            observation = game.get_frame(model)
            action = agent.get_action(observation)
            reward, observation2, terminal = game.feedback(action, model)
            if terminal:
                break

            agent.update(observation, action, reward, observation2, terminal)
            print("\n")
        lg.scalar_summary("episode-acc", game.performance, episode)
