import sys
sys.path.append('../')
import argparse
import torch
import gym
import numpy as np
from torch.autograd import Variable

from config import opt

def main():
    parser = argparse.ArgumentParser(description="-----[Agent tester]-----")
    parser.add_argument('--agent', default='dqn', help='Type of reinforcement agent. (dqn | policy, actor_critic)')
    parser.add_argument('--env', default='CartPole-v0', help='Type of reinforcement env.')
    params = parser.parse_args()

    env = gym.make(params.env)
    env = env.unwrapped

    opt.actions = env.action_space.n
    opt.state_size = env.observation_space.shape[0]
    opt.hidden_size = 8
    opt.batch_size_rl = 32
    opt.cuda = False
    opt.reward_clip = True
    opt.gamma = 0.99
    opt.data_sizes = [opt.state_size]
    opt.learning_rate_rl = 0.01

    from agents import DQNAgent, DQNTargetAgent, PolicyAgent, ActorCriticAgent, RandomAgent
    if params.agent == 'policy':
        agent = PolicyAgent()
    elif params.agent == 'dqn':
        agent = DQNAgent()
    elif params.agent == 'dqn_target':
        agent = DQNTargetAgent()
    elif params.agent == 'actor_critic':
        agent = ActorCriticAgent()
    elif params.agent == 'random':
        agent = RandomAgent()
    else:
        agent = DQNAgent()

    print('\nCollecting experience...')
    for i_episode in range(4000):
        state = env.reset()
        state = torch.FloatTensor(state).view(1, -1)
        score = 0
        done = False
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            next_state = torch.FloatTensor(next_state).view(1, -1)

            if not done:
                agent.update(state, action, r, next_state, done)

            score += 1
            state = next_state

            if done:
                agent.finish_episode(i_episode)
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(score, 2))
                break

if __name__ == "__main__":
    main()
