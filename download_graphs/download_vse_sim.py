from pprint import pprint
from visdom import Visdom
import pathlib
import json
import sys
import matplotlib.pyplot as plt

def download_vse_sim(env, env2):
    vis = Visdom('http://logserver.duckdns.org', port=5010)
    path = '../results/{}___{}'.format(env, env2)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    x_axis = []
    cap_recall_1 = []
    cap_recall_2 = []
    image_recall_1 = []
    image_recall_2 = []
    sum_1 = []
    sum_2 = []
    rsi = ['r1i', 'r5i', 'r10i']
    rs = ['r1', 'r5', 'r10']

    data = vis.get_window_data(env=env)
    d = json.loads(data)
    for key in d:
        try:
            x = list(d[key]["content"]["data"][0]["x"])
            y = list(d[key]["content"]["data"][0]["y"])
            if "/" in key and "avg_val" in key:
                match = key.split("/")[1]
                if match in rsi:
                    image_recall_1.append((y, match))
                    x_axis = x
                elif match in rs:
                    cap_recall_1.append((y, match))
                elif 'sum' in key:
                    sum_1 = (x, y)
        except:
            pass

    data = vis.get_window_data(env=env2)
    d = json.loads(data)
    for key in d:
        try:
            x = list(d[key]["content"]["data"][0]["x"])
            y = list(d[key]["content"]["data"][0]["y"])
            if "/" in key and "avg_val" in key:
                match = key.split("/")[1]
                if match in rsi:
                    image_recall_2.append((y, match))
                    x_axis = x
                elif match in rs:
                    cap_recall_2.append((y, match))
                elif 'sum' in key:
                    sum_2 = (x, y)
        except Exception as e:
            print(e)

    # Plot image recalls
    image_values_1 = [l[0] for l in image_recall_1]
    image_values_2 = [l[0] for l in image_recall_2]
    image_sum_1 = [sum(d) for d in zip(*image_values_1)]
    image_sum_2 = [sum(d) for d in zip(*image_values_2)]
    plots = [(image_sum_1, 'Random'), (image_sum_2, 'Intra caption')]
    plt.figure(1)
    plt.subplot(111)
    handles = []
    for (line, label) in plots:
        handle, = plt.plot(x_axis, line, label=label)
        handles.append(handle)
    plt.legend(handles=handles)
    plt.xlabel('Episodes')
    plt.ylabel('Image recall')
    plt.savefig('{}/image_recall.png'.format(path), dpi=600)

    # Plot caption recalls
    cap_values_1 = [l[0] for l in cap_recall_1]
    cap_values_2 = [l[0] for l in cap_recall_2]
    cap_sum_1 = [sum(d) for d in zip(*cap_values_1)]
    cap_sum_2 = [sum(d) for d in zip(*cap_values_2)]
    plots = [(cap_sum_1, 'Random'), (cap_sum_2, 'Intra caption')]
    plt.figure(2)
    plt.subplot(111)
    handles = []
    for (line, label) in plots:
        handle, = plt.plot(x_axis, line, label=label)
        handles.append(handle)
    plt.legend(handles=handles)
    plt.xlabel('Episodes')
    plt.ylabel('Caption recall')
    plt.savefig('{}/caption_recall.png'.format(path), dpi=600)

    # # Plot sum
    plots = [(sum_1[1], 'Random'), (sum_2[1], 'Intra caption')]
    plt.figure(3)
    plt.subplot(111)
    handles = []
    x_axis = sum_1[0]
    for (line, label) in plots:
        handle, = plt.plot(x_axis, line, label=label)
        handles.append(handle)
    plt.legend(handles=handles)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of recalls')
    plt.savefig('{}/sum.png'.format(path), dpi=600)


if __name__ == "__main__":
    env = sys.argv[1]
    env2 = sys.argv[2]
    download_vse_sim(env, env2)
