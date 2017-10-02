from model import CNN
import utils
import heapq, random
import logger
import datetime

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy

def to_np(x):
    return x.data.cpu().numpy()

def batchify(features, params):
    features = sorted(features, key=lambda x: len(x))
    # print(features)

    max_len = 0
    batch_matrix = []

    for feature in features:
        # print(feature)
        cur_len = 0

        for v_index, value in reversed(list(enumerate(feature))):
            # print(value.data[0])
            if value.data[0] != 21426:
                cur_len = v_index
                break
        max_len = max(cur_len, max_len)

    for feature in features:
        if params["CUDA"]:
            feature = feature.cuda(params["DEVICE"])

        if(len(feature) < max_len):
            padding = [21426 for x in range(max_len - len(feature))]
            feature.extend(padding)
            padding = torch.LongTensor(padding)

            if params["CUDA"]:
                padding = padding.cuda(params["DEVICE"])
            feature = torch.cat([feature, padding])
        else:
            feature = feature[0:max_len]
        batch_matrix.append(feature)

    batch_tensor = torch.stack(batch_matrix, dim=0)
    # print(batch_matrix)
    return batch_tensor
    # return torch.nn.utils.rnn.pack_padded_sequence(features, [len(x) for x in features])


def select_n_best_samples(model, data, params):
    if params["EVAL"]:
        model.eval()
    sample_scores = []
    completed = 0
    slide_n = 500
    print_every = 500
    sliding_scores = [0 for i in range(slide_n)]

    all_tensors = []
    all_targets = []

    max_len = 0

    for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

        batch_x = [[data["word_to_idx"][w] for w in sent] +
                   [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                   for sent in data["train_x"][i:i + batch_range]]
        batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]

        feature = Variable(torch.LongTensor(batch_x)).cuda(params["DEVICE"])
        # print(feature[1])
        target = Variable(torch.LongTensor(batch_y)).cuda(params["DEVICE"])
        all_tensors.extend(feature)
        all_targets.extend(target)

        if params["CUDA"]:
            feature, target = feature.cuda(params["DEVICE"]), target.cuda(params["DEVICE"])

        # print(feature)
        output = model(feature)

        output = torch.mul(output, torch.log(output))
        output = torch.sum(output, dim=1)
        output = output * -1
        # print(output)

        # l = len(target.data)
        # for s_index, score in enumerate([x for x in range(l)]):
        for s_index, score in enumerate(output):
            sample_scores.append(score.data[0])
            # sample_scores.append(0)
        completed += 1

        print("Selection process: {0:.0f}% completed ".format(
            100 * (completed / (len(data["train_x"]) // params["BATCH_SIZE"] + 1))), end="\r")

    best_n_indexes = [n[0] for n in heapq.nlargest(params["BATCH_SIZE"], enumerate(sample_scores), key=lambda x: x[1])]
    # best_n_indexes = [n[0] for n in random.sample(list(enumerate(sample_scores)), params["BATCH_SIZE"])]

    batch_features = []
    batch_target = []

    for index in best_n_indexes:
        batch_features.append(all_tensors[index])
        batch_target.append(all_targets[index].data[0])

    batch_feature = torch.stack(batch_features, dim=0)
    batch_target = torch.autograd.Variable(torch.LongTensor(batch_target))

    if params["CUDA"]:
        batch_feature = batch_feature.cuda(params["DEVICE"])
        batch_target = batch_target.cuda(params["DEVICE"])

    return batch_feature, batch_target, []

def train(data, params, lg):
    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    model = CNN(**params)
    if params["CUDA"]:
        model.cuda(params["DEVICE"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    max_test_acc = 0

    train_array = []

    data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

    for i in range(25):
        t1, t2, ret_array = select_n_best_samples(model, data, params)
        train_array.append((t1, t2))
        #
        print("\n")
        if params["RESET"]:
            model = CNN(**params)
            if params["CUDA"]:
                model.cuda(params["DEVICE"])

        print("Length of train set: {}".format(len(train_array)))
        for e in range(params["EPOCH"]):
            for feature, target in train_array:
                optimizer.zero_grad()
                model.train()
                pred = model(feature)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()

                # constrain l2-norms of the weight vectors
                if model.fc.weight.norm().data[0] > params["NORM_LIMIT"]:
                    model.fc.weight.data = model.fc.weight.data * params["NORM_LIMIT"] / model.fc.weight.data.norm()

        test(data, model, params, lg, i, mode="dev")

    best_model = {}
    return best_model

def test(data, model, params, lg, step, mode="test"):
    model.eval()
    if params["CUDA"]:
        model.cuda(params["DEVICE"])

    corrects, avg_loss = 0, 0
    for i in range(0, len(data["dev_x"]), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(data["dev_x"]) - i)

        feature = [[data["word_to_idx"][w] for w in sent] +
                   [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                   for sent in data["dev_x"][i:i + batch_range]]
        target = [data["classes"].index(c) for c in data["dev_y"][i:i + batch_range]]

        feature = Variable(torch.LongTensor(feature))
        target = Variable(torch.LongTensor(target))
        if params["CUDA"]:
            feature = feature.cuda(params["DEVICE"])
            target = target.cuda(params["DEVICE"])

        logit = model(feature)
        loss = torch.nn.functional.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data["dev_x"])
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    lg.scalar_summary("test-acc", accuracy, step+1)
    lg.scalar_summary("test-loss", avg_loss, step+1)
    for tag, value in model.named_parameters():
        if value.requires_grad:
            tag = tag.replace('.', '/')
            lg.histo_summary(tag, to_np(value), step+1)
            lg.histo_summary(tag+'/grad', to_np(value.grad), step+1)        
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})\n'.format(avg_loss,
                                                                  accuracy,
                                                                  corrects,
                                                                  size))
    # return accuracy    

def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="TREC", help="available datasets: MR, TREC")
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument("--save_model", default="F", help="whether saving model or not (T/F)")
    parser.add_argument("--early_stopping", default="F", help="whether to apply early stopping(T/F)")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.15, type=int, help="learning rate")
    parser.add_argument('--device', type=int, default=0, help='Cuda device to run on')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable the gpu' )
    parser.add_argument('--eval', action='store_true', default=False, help='Run eval() in selection')
    parser.add_argument('--reset', action='store_true', default=False, help='Reset model after each selection/train')

    options = parser.parse_args()
    data = getattr(utils, "read_{}".format(options.dataset))()

    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    print(options.learning_rate)

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": bool(options.save_model == "T"),
        "EARLY_STOPPING": bool(options.early_stopping == "T"),
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
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
        "EVAL": options.eval,
        "RESET": options.reset
    }
    params["CUDA"] = (not params["NO_CUDA"]) and torch.cuda.is_available(); del params["NO_CUDA"]

    lg = logger.Logger('./logs/cnn2/batch_size={},date={},FILTERS={},FILTER_NUM={},WORD_DIM={},MODEL={},DROPOUT_PROB={},NORM_LIMIT={},EVAL={},RESET={}'.format(
        params["BATCH_SIZE"],
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        str(params["FILTERS"]),
        str(params["FILTER_NUM"]),
        str(params["WORD_DIM"]),
        str(params["MODEL"]),
        str(params["DROPOUT_PROB"]),
        str(params["NORM_LIMIT"]),
        str(params["EVAL"]),
        str(params["RESET"])
        ))

    print("=" * 20 + "INFORMATION" + "=" * 20)
    for key, value in params.items():
        print("{}: {}".format(key.upper(), value))

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params, lg)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model = utils.load_model(params).cuda(params["DEVICE"])

        test_acc = test(data, model, params, -1, lg)
        print("test acc:", test_acc)

if __name__ == "__main__":
    main()
