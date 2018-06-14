import torch
import argparse
import datetime
import getpass
import importlib
# from data.vocab import Vocabulary  # NOQAimport logging
import os
import tensorboard_logger as tb_logger
import pickle
from config import opt, data, loaders, global_logger
from utils import external_logger, visdom_logger, local_logger, no_logger, load_word2vec
import uuid



def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train",
                        help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--save_model", default="F",
                        help="whether saving model or not (T/F)")
    parser.add_argument("--load_model_name", default="",
                        help="Name of the model to load from external server")
    parser.add_argument("--episodes", default=10000, type=int,
                        help="number of episodes")
    parser.add_argument("--hidden_size", default=4, type=int,
                        help="Size of hidden layer in deep RL")
    parser.add_argument("--learning_rate_rl", default=0.1,
                        type=float, help="learning rate")
    parser.add_argument('--data_path', default='/data/stud/jorgebjorn/data',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f8k_precomp',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--vocab_path', default='/data/stud/jorgebjorn/data/vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--init_samples', default=320, type=int,
                        help='number of random inital training data')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--batch_size_rl', default=32, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--budget', default=30, type=int,
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
                        help='Ensure the training is always done in train mode (Not recommended).')
    parser.add_argument('--log', default="no", help='Choose between: no, external, local, visdom')
    parser.add_argument('--no_cuda', action='store_true',
                        default=False, help='Disable cuda')
    parser.add_argument('--agent', default='dqn', help='Type of reinforcement agent. (dqn | policy, actor_critic)')
    parser.add_argument('--selection_radius', default=1, type=int, help='Selection radius')
    parser.add_argument('--topk', default=300, type=int, help='Topk similarity to use for state')
    parser.add_argument('--topk_image', default=0, type=int, help='Topk similarity images to use for state')
    parser.add_argument('--embedding', default='static', help='whether to pre-train a model and use its static embeddings or not. (static | train)')
    parser.add_argument('--image_distance', action='store_true', help='Include image distance in the state ')
    parser.add_argument('--reward_clip', action='store_true', help='Clip rewards using tanh')
    parser.add_argument('--val_size', default=500, type=int, help='Number of validation set size to use for reward')
    parser.add_argument('--train_shuffle', action='store_true', help='Shuffle active train set every time')
    parser.add_argument('--dataset', default='digit', help='Dataset. (vse | mr | digit)')
    parser.add_argument('--w2v', action='store_true', help='Use w2v embeddings')
    parser.add_argument('--c', default='', help='comment')
    parser.add_argument('--reward', default=2, help='minusreward')
    parser.add_argument("--gamma", default=0, type=float, help="Discount factor")
    parser.add_argument("--reward_threshold", default=1.0, type=float, help="Reward threshold")

    params = parser.parse_args()
    params.actions = 2

    params.logger_name = '{}_{}_{}_{}_{}'.format(getpass.getuser(), datetime.datetime.now().strftime("%d-%m-%y_%H:%M"), str(uuid.uuid4())[:4], params.agent, params.c)
    params.external_log_url = 'http://logserver.duckdns.org:5000'

    if torch.cuda.is_available():
        torch.cuda.set_device(params.device)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    # params.state_size = params.topk + params.topk_image + 1 if params.image_distance else params.topk + params.topk_image

    for arg in vars(params):
        opt[arg] = vars(params)[arg]

    if params.w2v:
        load_word2vec()

    # sending tensorboard logs to external server
    if params.log == "external":
        global_logger["lg"] = external_logger()

    # saving tensorboard logs local
    elif params.log == "local":
        global_logger["lg"] = local_logger()

    elif params.log == 'visdom':
        global_logger["lg"] = visdom_logger()
        global_logger["lg"].parameters_summary()

    # no logging at all, for testing purposes.
    else:
        global_logger["lg"] = no_logger()


    container = importlib.import_module('datasets.{}'.format(params.dataset))
    model = container.model
    load_data = container.load_data
    train_data, dev_data, test_data = load_data()
    data["train"] = train_data
    data["dev"] = dev_data
    data["test"] = test_data

    from train import train
    train(model)

if __name__ == "__main__":
    main()
