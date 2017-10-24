import utils
import logger
import datetime
import argparse
import torch

import train


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train",
                        help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="static",
                        help="available models: rand, static, non-static")
    parser.add_argument("--dataset", default="MR",
                        help="available datasets: MR, TREC")
    parser.add_argument('--batch-size', type=int, default=25,
                        help='batch size for training [default: 25]')
    parser.add_argument("--save_model", default="F",
                        help="whether saving model or not (T/F)")
    parser.add_argument("--early_stopping", default="F",
                        help="whether to apply early stopping(T/F)")
    parser.add_argument("--epoch", default=100, type=int,
                        help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.1,
                        type=int, help="learning rate")
    parser.add_argument('--device', type=int, default=0,
                        help='Cuda device to run on')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disable the gpu')
    parser.add_argument("--scorefn", default="entropy",
                        help="available scoring functions: entropy, rand, egl")
    parser.add_argument('--average', type=int, default=1,
                        help='Number of runs to average [default: 1]')

    parser.add_argument('--hidden', type=int, default=300,
                        help='Size of the hidden layer')

    options = parser.parse_args()
    data = getattr(utils, "read_{}".format(options.dataset))()

    data["vocab"] = sorted(list(set(
        [w for sent in data["train_x"] + data["dev_x"]
            + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": bool(options.save_model == "T"),
        "EARLY_STOPPING": bool(options.early_stopping == "T"),
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
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "DEVICE": options.device,
        "NO_CUDA": options.no_cuda,
        "SCORE_FN": options.scorefn,
        "N_AVERAGE": options.average,
        "HIDDEN_SIZE": options.hidden
    }

    params["CUDA"] = (not params["NO_CUDA"]) and torch.cuda.is_available()
    del params["NO_CUDA"]

    lg = logger.Logger('./logs/cnn2/batch_size={},date={},FILTERS={},FILTER_NUM={},WORD_DIM={},MODEL={},DROPOUT_PROB={},NORM_LIMIT={},SCORE_FN={},AVERAGE={}'.format(
        params["BATCH_SIZE"],
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        str(params["FILTERS"]),
        str(params["FILTER_NUM"]),
        str(params["WORD_DIM"]),
        str(params["MODEL"]),
        str(params["DROPOUT_PROB"]),
        str(params["NORM_LIMIT"]),
        str(params["SCORE_FN"]),
        str(params["N_AVERAGE"])
    ))

    print("=" * 20 + "INFORMATION" + "=" * 20)
    for key, value in params.items():
        print("{}: {}".format(key.upper(), value))

    if options.mode == "train":

        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        # model = train.active_train(data, params, lg)
        train.active_train(data, params, lg)
        # if params["SAVE_MODEL"]:
        #     utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model = utils.load_model(params).cuda(params["DEVICE"])

        test_acc = train.evaluate(data, model, params, -1, lg)
        print("test acc:", test_acc)


if __name__ == "__main__":
    main()
