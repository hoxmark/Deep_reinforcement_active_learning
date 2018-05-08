import argparse
import torch
import gym
from torch.autograd import Variable

from config import opt

def main():
    parser = argparse.ArgumentParser(description="-----[Agent tester]-----")
    parser.add_argument('--agent', default='dqn_target', help='Type of reinforcement agent. (dqn | policy, actor_critic)')
    parser.add_argument('--env', default='CartPole-v0', help='Type of reinforcement env.')
    params = parser.parse_args()

    env = gym.make(params.env)
    env = env.unwrapped

    opt.actions = env.action_space.n
    opt.state_size = env.observation_space.shape[0]
    opt.hidden_size = 50
    opt.cuda = False
    opt.reward_clip = True

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
        agent = DQNTargetAgent()

    print('\nCollecting experience...')
    for i_episode in range(4000):
        s = env.reset()
        s = Variable(torch.FloatTensor(s)).view(1, -1)
        ep_r = 0
        while True:
            env.render()
            a = int(agent.get_action(s))
            # take action
            s_, r, done, info = env.step(a)
            # modify the reward
            if params.env == 'CartPole-v0':
                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2
            s_ = Variable(torch.FloatTensor(s_)).view(1, -1)
            agent.update(s, a, r, s_, done)

            ep_r += r

            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
                agent.finish_episode(i_episode)
                break
            s = s_

if __name__ == "__main__":
    main()
