
# Active Deep Learning for Sentence Classification using CNN and RNN in pytorch

This is the implementation of "Active Discriminative Text Representation Learning" (http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14174/14265) with modification. it is implmented using **Pytorch**.


## Results

Below are results corresponding to RNN and CNN using out 3 different selection scores, random, entropy and EGL. We are using a dataset with 500 samples.

(Measure: Accuracy)

| Model        | Selection score    | MR        | TREC |
|--------------|:------------------:|:---------:|:----:|
| CNN          | Random             | 73.29     |       | 
|              | Entropy            | 74.57     |       |
|              | EGL                | **76.80** |    31.82   |
| RNN          | Random             | 72.60     |       |
|              | Entropy            | 75.87     |       |
|              | EGL                | **77.77** |   28.86    |



## Development Environment
- OS: Ubuntu Ubuntu 16.04.2 LTS (64bit)
- Language: Python 3.5.2
- GPU: 2xTesla P100


## Requirements

This model is based on pre-trained Word2vec([GoogleNews-vectors-negative300.bin](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)) by T.Mikolov et al.
You should download this file and place it in the root folder.

Also you should follow library requirements specified in the **requirements.txt**.

    backports.shutil-get-terminal-size==1.0.0
    bleach==1.5.0
    boto==2.48.0
    bz2file==0.98
    certifi==2017.7.27.1
    chardet==3.0.4
    decorator==4.1.2
    gensim==2.3.0
    html5lib==0.9999999
    idna==2.6
    ipython-genutils==0.2.0
    jsonschema==2.6.0
    jupyter-core==4.4.0
    Markdown==2.6.9
    nbformat==4.4.0
    numpy==1.13.3
    pkg-resources==0.0.0
    plotly==2.2.2
    protobuf==3.4.0
    pytz==2017.3
    PyYAML==3.12
    reprint==0.5.0.1
    requests==2.18.4
    scikit-learn==0.19.0
    scipy==0.19.1
    six==1.11.0
    smart-open==1.5.3
    tensorboardX==0.8
    tensorflow==1.3.0
    tensorflow-tensorboard==0.1.7
    torch==0.2.0.post3
    traitlets==4.3.2
    urllib3==1.22
    Werkzeug==0.12.2


## Execution

> usage: main.py [-h] [--mode MODE] [--model MODEL] [--embedding EMBEDDING]
               [--dataset DATASET] [--batch-size BATCH_SIZE]
               [--selection-size SELECTION_SIZE] [--save_model SAVE_MODEL]
               [--early_stopping EARLY_STOPPING] [--epoch EPOCH]
               [--learning_rate LEARNING_RATE] [--dropout_embed DROPOUT_EMBED]
               [--dropout_model DROPOUT_MODEL] [--device DEVICE] [--no-cuda]
               [--scorefn SCOREFN] [--average AVERAGE] [--hnodes HNODES]
               [--hlayers HLAYERS] [--weight_decay WEIGHT_DECAY] [--no-log]
               [--minibatch]

-----[CNN-classifier]-----

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           train: train (with test) a model / test: test saved
                        models
  --model MODEL         Type of model to use. Default: CNN. Available models:
                        CNN, RNN
  --embedding EMBEDDING
                        available embedings: random, static
  --dataset DATASET     available datasets: MR, TREC
  --batch-size BATCH_SIZE
                        batch size for training [default: 25]
  --selection-size SELECTION_SIZE
                        selection size for selection function [default: 25]
  --save_model SAVE_MODEL
                        whether saving model or not (T/F)
  --early_stopping EARLY_STOPPING
                        whether to apply early stopping(T/F)
  --epoch EPOCH         number of max epoch
  --learning_rate LEARNING_RATE
                        learning rate
  --dropout_embed DROPOUT_EMBED
                        Dropout embed probability. Default: 0.2
  --dropout_model DROPOUT_MODEL
                        Dropout model probability. Default: 0.4
  --device DEVICE       Cuda device to run on
  --no-cuda             disable the gpu
  --scorefn SCOREFN     available scoring functions: entropy, random, egl
  --average AVERAGE     Number of runs to average [default: 1]
  --hnodes HNODES       Number of nodes in the hidden layer(s)
  --hlayers HLAYERS     Number of hidden layers
  --weight_decay WEIGHT_DECAY
                        Value of weight_decay
  --no-log              Disable logging
  --minibatch           Use minibatch training