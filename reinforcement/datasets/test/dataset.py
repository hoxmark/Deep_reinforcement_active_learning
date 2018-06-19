import random
import torch

from config import opt

def load_data():
    num_data = 10000
    softmax_scale_factor = 10
    x = []
    y = []
    opt.state_size = 2

    max_reward = torch.Tensor([1/opt.state_size for i in range(0, opt.state_size)])
    max_reward = torch.mul(max_reward, torch.log(max_reward))
    max_reward = torch.sum(max_reward)
    max_reward = max_reward * -1

    for i in range(num_data):
        probs = [random.random() * softmax_scale_factor for i in range(opt.state_size)]
        probs = torch.Tensor(probs)
        probs = torch.nn.functional.softmax(probs, dim=0)
        probs = probs.sort()[0]
        x.append(probs.cpu().numpy())

        # Calculate entropy
        reward = torch.mul(probs, torch.log(probs))
        reward = torch.sum(reward)
        reward = reward * -1

        # Scale with max reward to get it in range [0, 1]
        reward = (reward / max_reward) - 0.6
        y.append(reward.data.item())

    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9
    train_data = (x[:dev_idx], y[:dev_idx])
    dev_data = (x[dev_idx:test_idx], y[dev_idx:test_idx])
    test_data = (x[test_idx:], y[test_idx:])
    opt.data_len = len(train_data[0])

    return train_data, dev_data, test_data
