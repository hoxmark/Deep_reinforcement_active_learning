from pprint import pprint
from visdom import Visdom
import pathlib
import json
import sys
import matplotlib.pyplot as plt

def download_env(env):
    vis = Visdom('http://logserver.duckdns.org', port=5010)
    data = vis.get_window_data(env=env)
    d = json.loads(data)
    path = './results/{}'.format(env)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    x_axis = []
    sum = []
    num_0_actions = []
    cap_recall = []
    image_recall = []
    rsi = ['r1i', 'r5i', 'r10i']
    rs = ['r1', 'r5', 'r10']

    for key in d:
        try:
            x = list(d[key]["content"]["data"][0]["x"])
            y = list(d[key]["content"]["data"][0]["y"])
            if any(r in key for r in rsi):
                x_axis = x
                image_recall.append(y)
            elif any(r in key for r in rs):
                cap_recall.append(y)
            elif 'sum' in key:
                sum = (x, y)
            elif 'actions' in key:
                num_0_actions = (x,y)

        except:
            pass

    plt.figure(1)
    plt.subplot(111)
    for line in image_recall:
        plt.plot(x_axis, line)
    plt.savefig('results/{}/image_recall.svg'.format(env))

    plt.figure(2)
    plt.subplot(111)
    plt.plot(*sum)
    plt.savefig('results/{}/sum.svg'.format(env))

if __name__ == "__main__":
    env = sys.argv[1]
    download_env(env)
