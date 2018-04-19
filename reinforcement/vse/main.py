import datetime
import argparse
import torch
import getpass
import logging
import pickle
import os
import tensorboard_logger as tb_logger

from train import train
from config import opt, data, loaders, global_logger
from data.evaluation import encode_data
from data.utils import external_logger, local_logger, no_logger
from data.vocab import Vocabulary  # NOQA
from data.dataset import get_loaders


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train",
                        help="train: train (with test) a model / test: test saved models")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training [default: 32]')
    parser.add_argument("--save_model", default="F",
                        help="whether saving model or not (T/F)")
    parser.add_argument("--load_model_name", default="",
                        help="Name of the model to load from external server")
    parser.add_argument("--episodes", default=10000, type=int,
                        help="number of episodes")
    parser.add_argument("--learning_rate_rl", default=0.1,
                        type=float, help="learning rate")
    parser.add_argument('--data_path', default='/data/stud/jorgebjorn/data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f8k_precomp',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--vocab_path', default='/data/stud/jorgebjorn/data/vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=5, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--budget', default=150, type=int,
                        help='Our labeling budget')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate_vse', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='/data/stud/jorgebjorn/runs/{}/{}'.format(getpass.getuser(), datetime.datetime.now().strftime("%d-%m-%y_%H:%M")),
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--device', default=0, type=int,
                        help='which gpu to use')
    parser.add_argument('--cnn_type', default='vgg19',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true',
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')
    parser.add_argument('--log', default="no", help='Choose between: no, external, local')
    parser.add_argument('--no_cuda', action='store_true',
                        default=False, help='Disable cuda')
    parser.add_argument('--agent', default='dqn', help='Type of reinforcement agent. (dqn | policy)')

    params = parser.parse_args()
    params.actions = 2
    params.logger_name = '{}_{}'.format(datetime.datetime.now().strftime("%d-%m-%y_%H:%M"), params.agent)
    params.external_log_url = 'http://logserver.duckdns.org:5001'

    if torch.cuda.is_available():
        torch.cuda.set_device(params.device)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()

    vocab = pickle.load(open(os.path.join(params.vocab_path, '%s_vocab.pkl' % params.data_name), 'rb'))
    params.vocab = vocab
    params.vocab_size = len(vocab)

    active_loader, train_loader, val_loader, val_tot_loader = get_loaders(
        params.data_name, vocab, params.crop_size, params.batch_size, params.workers, params)

    loaders["active_loader"] = active_loader
    loaders["train_loader"] = train_loader
    loaders["val_loader"] = val_loader          #limited val dataset
    loaders["val_tot_loader"] = val_tot_loader  #Total val dataset for validation each episode
    # TODO Check if this is correct order

    for arg in vars(params):
        opt[arg] = vars(params)[arg]

    # sending tensorboard logs to external server
    if params.log == "external":
        global_logger["lg"] = external_logger()

    # saving tensorboard logs local
    elif params.log == "local":
        global_logger["lg"] = local_logger()

    # no logging at all, for testing purposes.
    else:
        global_logger["lg"] = no_logger()

    train()

if __name__ == "__main__":
    main()
