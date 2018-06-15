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
    opt.batch_size_rl = 64
    opt.cuda = False
    opt.reward_clip = True
    opt.gamma = 0

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
        state = env.reset()
        state = Variable(torch.FloatTensor(state)).view(1, -1)
        score = 0
        done = False
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = Variable(torch.FloatTensor(next_state)).view(1, -1)

            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -10
            agent.update(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                agent.finish_episode(i_episode)
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(score, 2))
                break

if __name__ == "__main__":
    main()
