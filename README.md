
# Repo for our master thesis

This is the implementation of "Active Discriminative Text Representation Learning" (http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14174/14265) with modification. it is implmented using **Pytorch**.

This model is based on pre-trained Word2vec([GoogleNews-vectors-negative300.bin](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)) by T.Mikolov et al.


## Results

Below are results corresponding to RNN and CNN using out 3 different selection scores, random, entropy and EGL. We are using a dataset with 500 samples.

(Measure: Accuracy)

| Model        | Selection score    | MR        | TREC  |
|--------------|:------------------:|:---------:|:-----:|
| CNN          | Random             | 73.29     |82.05  |
|              | Entropy            | 74.57     |82.82  |
|              | EGL                | **76.80** |79.30  |
| RNN          | Random             | 72.60     |78.07  |
|              | Entropy            | 75.87     |75.65  |
|              | EGL                | **77.77** |74.0   |



## Development Environment
- OS: Ubuntu Ubuntu 16.04.2 LTS (64bit)
- Language: Python 3.5.2
- GPU: 2xTesla P100



## Installing and running the project

1. Clone this github repo to you machine.

2.  Download [GoogleNews-vectors-negative300.bin] and place it in the root folder.

```sh
$ wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
$ gunzip GoogleNews-vectors-negative300.bin.gz
```

3.  Install pytorch, we are running python 3.5.2 and cuda so we used the following command:

```sh
$ pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
```

  if you are using a different version, we do not know if it will work as intended, head over to http://pytorch.org to download it.

4. I nstall all the rquired python dependecies using pip3.

```sh
$ pip3 install -r /path/to/requirements.txt
```

## Execution
TODO: you need to set the correct data_path.

Example:
```sh
$ python  main.py  --embedding static --scorefn entropy --model cnn --average 10
```

The help commpand if you want to run it with different hyperparamters.

```sh
$ python  main.py  --help

usage: main.py
  [-h]
  [--mode MODE]
  [--model MODEL]
  [--embedding EMBEDDING]
  [--dataset DATASET]
  [--batch-size BATCH_SIZE]
  [--selection-size SELECTION_SIZE]
  [--save_model SAVE_MODEL]
  [--early_stopping EARLY_STOPPING]
  [--epoch EPOCH]
  [--learning_rate LEARNING_RATE]
  [--dropout_embed DROPOUT_EMBED]
  [--dropout_model DROPOUT_MODEL]
  [--device DEVICE]
  [--no-cuda]
  [--scorefn SCOREFN]
  [--average AVERAGE]
  [--hnodes HNODES]
  [--hlayers HLAYERS]
  [--weight_decay WEIGHT_DECAY]
  [--no-log]
  [--minibatch]

...
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
  --minibatch           Use minibatch training, default true
  ...

```
