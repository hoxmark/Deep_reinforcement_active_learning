import os
import pprint
import atexit
import json
import torch
import torch.nn as nn
import tensorflow as tf
import pickle


from pprint import pprint
from flask import Flask, render_template, request

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

app = Flask(__name__)
base_dir = '/home/public'
log_dir = '{}/logs'.format(base_dir)
model_dir = '{}/models'.format(base_dir)
print(log_dir)
print(model_dir)


# Save Logs: Only works for scalar summary
@app.route('/post_log/<name>', methods=['POST'])
def post_log(name):
    content = request.get_json(silent=True)
    writer = tf.summary.FileWriter('{}/{}'.format(log_dir, name))
    summary = tf.Summary(value=[tf.Summary.Value(
        tag=content["tag"], simple_value=content["value"])])
    writer.add_summary(summary, content["step"])
    print(content)
    writer.flush()
    writer.close()
    return json.dumps(content)

# Save model
@app.route('/save_model/<agent>/<episode>/<name>', methods=['POST'])
def save_model(agent, episode, name):
    pkl = pickle.loads(request.data)
    filename = '{}/{}/{}/{}.pkl'.format(model_dir, agent, episode, name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(pkl, f)
    return "success"

# Load model
@app.route('/load_model/<agent>/<episode>/<name>', methods=['GET'])
def load_model(agent, episode, name):
    filename = '{}/{}/{}/{}.pkl'.format(model_dir, agent, episode, name)
    pkl = pickle.load(open(filename, "rb"))
    return pickle.dumps(pkl)


# Not yet in use
# Save chosen parameters
@app.route('/post_params/<name>', methods=['POST'])
def post_params(name):
    content = request.get_json(silent=True)
    print(content)
    with open('{}/{}/parameters.json'.format(log_dir, name), 'w') as outfile:
        json.dump(content, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    return json.dumps(content)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
