import pprint
import atexit
import tensorflow as tf
import json

from pprint import pprint
from flask import Flask, render_template, request

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

app = Flask(__name__)

# Only works for scalar summary
@app.route('/post_log/<logdir>', methods=['POST'])
def post_log(logdir):
    content = request.get_json(silent=True)
    writer = tf.summary.FileWriter('/home/jorgenwilhelmsen/logs/{}'.format(logdir))
    summary = tf.Summary(value=[tf.Summary.Value(tag=content["tag"], simple_value=content["value"])])
    writer.add_summary(summary, content["step"])
    print(content)
    writer.flush()
    writer.close()
    return json.dumps(content)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
