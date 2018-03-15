import utils
import logger
import datetime
import argparse
import torch

import train
from config import params, data
from models import rnnae


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--embedding", default="w2v",
                        help="available embedings: random, w2v")
    parser.add_argument("--dataset", default="MR",
                        help="available datasets: MR, TREC")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training [default: 32]')
    parser.add_argument("--epoch", default=100, type=int,
                        help="number of max epoch")
    parser.add_argument("--learning_rate", default=1e-3,
                        type=float, help="learning rate")
    parser.add_argument("--dropout_embed", default=0.2,
                        type=float, help="Dropout embed probability. Default: 0.2")
    parser.add_argument('--device', type=int, default=0,
                        help='Cuda device to run on')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disable the gpu')
    parser.add_argument('--hsize', type=int, default=256,
                        help='Number of nodes in the hidden layer(s)')
    parser.add_argument('--hlayers', type=int, default=1,
                        help='Number of hidden layers')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Value of weight_decay')
    parser.add_argument('--no-log', action='store_true',
                        default=False, help='Disable logging')

    options = parser.parse_args()


    getattr(utils, "read_{}".format(options.dataset))()

    data["vocab"] = sorted(list(set(
        [w for sent in data["train_x"] + data["dev_x"]
            + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}

    params_local = {
        "DATASET": options.dataset,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"]
                             + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": options.batch_size,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_EMBED": options.dropout_embed,
        "DEVICE": options.device,
        "NO_CUDA": options.no_cuda,
        "HIDDEN_SIZE": options.hsize,
        "HIDDEN_LAYERS": options.hlayers,
        "WEIGHT_DECAY": options.weight_decay,
        "LOG": not options.no_log
    }

    for key in params_local:
        params[key] = params_local[key]

    params["CUDA"] = (not params["NO_CUDA"]) and torch.cuda.is_available()
    del params["NO_CUDA"]

    if params["CUDA"]:
        torch.cuda.set_device(params["DEVICE"])

    encoder = rnnae.EncoderRNN()
    decoder = rnnae.AttnDecoderRNN()

    if params["CUDA"]:
        encoder, decoder = encoder.cuda(), decoder.cuda()

    rnnae.train(encoder, decoder)
    torch.save(encoder.state_dict(), "./saved_models/rnn-encoder-{}-{}".format(params["DATASET"], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    torch.save(decoder.state_dict(), "./saved_models/rnn-decoder-{}-{}".format(params["DATASET"], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

if __name__ == "__main__":
    main()
