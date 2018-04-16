import pprint
import atexit
import pytz
from pprint import pprint
import json

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
import tensorflow as tf

from flask import Flask, render_template, request

app = Flask(__name__)
@app.route("/")
def index():
    return render_template('index.html')


# Only works for scalar summary
@app.route('/post_log/<logdir>', methods=['POST'])
def post_log(logdir):
    content = request.get_json(silent=True)
    # print(logdir)
    # print(content)
    # return 'hei'

    writer = tf.summary.FileWriter(logdir)
    summary = tf.Summary(value=[tf.Summary.Value(tag=content.tag, simple_value=content.value)])
    writer.add_summary(summary, content.step)
    print(content)
    return json.dumps(content)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
