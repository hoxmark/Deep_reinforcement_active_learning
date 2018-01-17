from agent import RobotCNNDQN
from models.cnn import CNN
from game import Game

def train(data, params):
    agent = RobotCNNDQN(params)
    model = CNN(data, params)

    game = Game(data, params)

    for episode in range(params["EPISODES"]): 
        print '>>>>>>> Current game round ', episode, 'Maximum ', params["EPISODES"]
        observation = game.get_frame(model)
        action = robot.get_action(observation)
        print '> Action', action
        reward, observation2, terminal = game.feedback(action, model)
        print '> Reward', reward
        robot.update(observation, action, reward, observation2, terminal)
        if terminal == True:
            episode += 1
            print '> Terminal <'
