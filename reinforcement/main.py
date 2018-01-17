import utils
import logger
import datetime
import argparse
import torch

from train import train


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train",
                        help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="cnn",
                        help="Type of model to use. Default: CNN. Available models: CNN, RNN")
    parser.add_argument("--embedding", default="static",
                        help="available embedings: random, static")
    parser.add_argument("--dataset", default="MR",
                        help="available datasets: MR, TREC")
    parser.add_argument('--batch-size', type=int, default=25,
                        help='batch size for training [default: 25]')
    parser.add_argument("--save_model", default="F",
                        help="whether saving model or not (T/F)")
    parser.add_argument("--episodes", default=100, type=int,
                        help="number of episodes")
    parser.add_argument("--learning_rate", default=0.1,
                        type=float, help="learning rate")
    parser.add_argument("--dropout_embed", default=0.2,
                        type=float, help="Dropout embed probability. Default: 0.2")
    parser.add_argument("--dropout_model", default=0.4,
                        type=float, help="Dropout model probability. Default: 0.4")
    parser.add_argument('--device', type=int, default=0,
                        help='Cuda device to run on')
    parser.add_argument('--average', type=int, default=1,
                        help='Number of runs to average [default: 1]')
    parser.add_argument('--hnodes', type=int, default=128,
                        help='Number of nodes in the hidden layer(s)')
    parser.add_argument('--hlayers', type=int, default=1,
                        help='Number of hidden layers')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Value of weight_decay')
    parser.add_argument('--no-log', action='store_true',
                        default=False, help='Disable logging')

    options = parser.parse_args()
    data = getattr(utils, "read_{}".format(options.dataset))()

    data["vocab"] = sorted(list(set(
        [w for sent in data["train_x"] + data["dev_x"]
            + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "EMBEDDING": options.embedding,
        "DATASET": options.dataset,
        "SAVE_MODEL": bool(options.save_model == "T"),
        "EPISODES": options.episodes,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"]
                             + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": options.batch_size,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "ACTIONS": 2,
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_EMBED": options.dropout_embed,
        "DROPOUT_MODEL": options.dropout_model,
        "DEVICE": options.device,
        "N_AVERAGE": options.average,
        "HIDDEN_SIZE": options.hnodes,
        "HIDDEN_LAYERS": options.hlayers,
        "WEIGHT_DECAY": options.weight_decay,
        "LOG": not options.no_log
    }

    params["CUDA"] = (not params["NO_CUDA"]) and torch.cuda.is_available()
    del params["NO_CUDA"]

    if params["CUDA"]:
        torch.cuda.set_device(params["DEVICE"])

    print("=" * 20 + "INFORMATION" + "=" * 20)
    for key, value in params.items():
        print("{}: {}".format(key.upper(), value))

    print("=" * 20 + "TRAINING STARTED" + "=" * 20)
    # train.active_train(data, params)
    train(data, params)
    print("=" * 20 + "TRAINING FINISHED" + "=" * 20)


if __name__ == "__main__":
    main()
