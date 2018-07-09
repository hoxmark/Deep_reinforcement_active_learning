from pprint import pprint
from visdom import Visdom
import pathlib
import json
import sys
import matplotlib.pyplot as plt

def download_vse_env(env):
    vis = Visdom('http://logserver.duckdns.org', port=5010)
    data = vis.get_window_data(env=env)
    d = json.loads(data)
    path = '../results/{}'.format(env)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    x_axis = []
    sum = []
    loss = []
    num_0_actions = []
    cap_recall = []
    image_recall = []
    rsi = ['r1i', 'r5i', 'r10i']
    rs = ['r1', 'r5', 'r10']

    # TODO idk. Dirty fix
    added_cap_labels = []
    for key in d:
        try:
            x = list(d[key]["content"]["data"][0]["x"])
            y = list(d[key]["content"]["data"][0]["y"])
            if "/" in key:
                match = key.split("/")[1]
                if match in rsi:
                    image_recall.append((y, match))
                    x_axis = x
                elif match in rs:
                    cap_recall.append((y, match))
                elif 'sum' in key:
                    sum = (x, y)
            elif 'actions' in key:
                num_0_actions = (x,y)
                loss = (x, y)
        except:
            pass
    # cap_recall = sorted(cap_recall, key=lambda x: x[1])
    # cap_recall = sorted(cap_recall, key=lambda x: x[1])

    # Plot image recalls
    plt.figure(1)
    plt.subplot(111)
    handles = []
    for (line, label) in image_recall:
        handle, = plt.plot(x_axis, line, label=label)
        handles.append(handle)
    plt.legend(handles=handles)
    plt.xlabel('Episodes')
    plt.ylabel('Image recall')
    plt.savefig('{}/image_recall.png'.format(env), dpi=600)

    # Plot caption recalls
    plt.figure(2)
    plt.subplot(111)
    handles = []
    for (line, label) in cap_recall:
        handle, = plt.plot(x_axis, line, label=label)
        handles.append(handle)
    plt.legend(handles=handles)
    plt.xlabel('Episodes')
    plt.ylabel('Caption recall')
    plt.savefig('{}/caption_recall.png'.format(path, env), dpi=600)

    # Plot sum
    plt.figure(3)
    plt.subplot(111)
    plt.plot(*sum)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of recalls')
    plt.savefig('{}/sum.png'.format(path, env), dpi=600)

    # Plot loss
    plt.figure(4)
    plt.subplot(111)
    plt.plot(*loss)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.savefig('{}/loss.png'.format(path, env), dpi=600)

    # Plot sum
    plt.figure(5)
    plt.subplot(111)
    plt.plot(*num_0_actions)
    plt.xlabel('Episodes')
    plt.ylabel('Number of 0 actions')
    plt.savefig('{}/num_0_actions.png'.format(path, env), dpi=600)

if __name__ == "__main__":
    env = sys.argv[1]
    download_vse_env(env)
