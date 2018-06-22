import sklearn
from config import opt, data

def load_data():
    x, y = [], []
    with open("{}/MR/rt-polarity.pos".format(opt.data_path), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("{}/MR/rt-polarity.neg".format(opt.data_path), "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = sklearn.utils.shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data.vocab = sorted(list(set([w for sent in x for w in sent])))
    data.word_to_idx = {w: i for i, w in enumerate(data.vocab)}

    x_pad = []
    for sentence in x:
        tokens = [data.word_to_idx[word] for word in sentence]
        padding = (59 - len(sentence)) * [len(data.vocab) + 1]
        tokens_padded = tokens + padding
        x_pad.append(tokens_padded)

    train_data = (x_pad[:dev_idx], y[:dev_idx])
    dev_data = (x_pad[dev_idx:test_idx], y[dev_idx:test_idx])
    test_data = (x_pad[test_idx:], y[test_idx:])
    opt.state_size = 2
    opt.data_len = len(train_data[0])

    return train_data, dev_data, test_data
