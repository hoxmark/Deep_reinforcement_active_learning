from pprint import pprint
from visdom import Visdom
import pathlib
import json
import sys

def download_env(env):
    vis = Visdom('http://logserver.duckdns.org', port=5010)
    data = vis.get_window_data(env=env)
    d = json.loads(data)
    path = './results/{}'.format(env)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    for key in d:
        with open('{}/{}.csv'.format(path, key.replace('/', '_')),'w+') as file:
            try:
                x = d[key]["content"]["data"][0]["x"]
                y = d[key]["content"]["data"][0]["y"]

                file.write('Step, Value \n')
                for step, value in zip(x, y):
                    file.write("{}, {}\n".format(step, value))
                print(key)
            except:
                pass

if __name__ == "__main__":
    env = sys.argv[1]
    download_env(env)
