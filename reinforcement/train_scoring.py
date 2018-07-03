import os
import random
import copy
import torch
import itertools
import numpy as np
from game import Game
from agents import DQNAgent, DQNTargetAgent, PolicyAgent, ActorCriticAgent, RandomAgent
from config import data, opt, loaders, global_logger
from utils import save_model, timer, load_external_model, average_vector, save_VSE_model,get_full_VSE_model, pairwise_distances


def active_train(classifier):
    lg = global_logger["lg"]
    model = classifier()

    if opt.scorefn == 'random':
        scorefn = random_scorefn
    else:
        scorefn = intra_scorefn
    n_rounds = 20
    avg_scores = [{} for i in range(n_rounds)]

    for avg_i in range(opt.n_average):
        # Start with all the data, reset active set
        data["train_deleted"] = copy.deepcopy(data["train"])
        data["active"] = tuple(([] for i in range(len(data["train"]))))

        # Init validation
        model.reset()
        indices = random_scorefn(model, 1120)
        for idx in indices:
            model.add_index(idx)
        # Delete the data from train_deleted
        new_data = [*data["train_deleted"]]
        for i, d in enumerate(new_data):
            new_data[i] = np.delete(new_data[i], indices, axis=0)
        data["train_deleted"] = new_data
        model.train(data["active"])

        metrics = model.validate(data["dev"])
        for key in metrics:
            if key in avg_scores[rnd]:
                avg_scores[rnd][key].append(metrics[key])
            else:
                avg_scores[rnd][key] = [metrics[key]]

        for rnd in range(1, n_rounds):

            # Get and add indices according to scorefn
            indices = scorefn(model)
            for idx in indices:
                model.add_index(idx)
            # Delete the data from train_deleted
            new_data = [*data["train_deleted"]]
            for i, d in enumerate(new_data):
                new_data[i] = np.delete(new_data[i], indices, axis=0)
            data["train_deleted"] = new_data
            print(len(data["train_deleted"][0]))

            # Reset and train model
            model.reset()
            timer(model.train_model, (data["active"], opt.num_epochs))
            metrics = model.validate(data["dev"])
            lg.dict_scalar_summary('last_episode_validation', metrics, rnd-1)

            for key in metrics:
                if key in avg_scores[rnd]:
                    avg_scores[rnd][key].append(metrics[key])
                else:
                    avg_scores[rnd][key] = [metrics[key]]
                print("adding {} at round {}".format(key, rnd))
                print(len(avg_scores[rnd][key]))

        for id, round_metric in enumerate(avg_scores):
            for key in round_metric:
                tag = 'avg_val/{}'.format(key)
                avg = sum(round_metric[key]) / len(round_metric[key])
                lg.scalar_summary(tag, avg, id*5*32)


def random_scorefn(model, n_samples=160):
    dataset = data["train_deleted"]
    indices = random.sample(list(range(0, len(dataset[0]))), n_samples)
    return indices

def intra_scorefn(model):
    """ Encodes data from data["train"] to use in the episode calculations """
    dataset = data["train_deleted"]
    torch.set_grad_enabled(False)
    img_embs, cap_embs = timer(model.encode_data, (dataset,))
    if opt.cuda:
        img_embs = img_embs.cuda()
        cap_embs = cap_embs.cuda()
    image_caption_distances = timer(pairwise_distances, (img_embs, cap_embs))
    topk = torch.topk(image_caption_distances, opt.topk, 1, largest=False)
    (image_caption_distances_topk, image_caption_distances_topk_idx) = (topk[0], topk[1])
    # data["image_caption_distances_topk"] = image_caption_distances_topk
    data["image_caption_distances_topk_idx"] = image_caption_distances_topk_idx
    del topk
    del image_caption_distances
    intra_cap_distance = timer(pairwise_distances, (cap_embs, cap_embs))
    select_indices_row = []
    select_indices_col = []

    for row in data["image_caption_distances_topk_idx"].cpu().numpy():
        permutations = list(zip(*itertools.permutations(row, 2)))
        permutations_list = [list(p) for p in permutations]
        select_indices_row.extend(permutations_list[0])
        select_indices_col.extend(permutations_list[1])

    all_dist = intra_cap_distance[select_indices_row, select_indices_col]
    all_dist = all_dist.view(len(data["train_deleted"][0]), opt.topk, opt.topk -1)
    all_dist = all_dist.mean(dim=2).mean(dim=1)
    indices = torch.topk(all_dist, 32 * 5, 0, largest=True)[1]
    print(len(indices))

    # print(all_dist.size())
    # data["all_states"] = torch.cat((torch.Tensor(data["train"][0]), all_dist.cpu(), data["image_caption_distances_topk"].cpu()), dim=1).cpu()
    del intra_cap_distance
    del img_embs
    del cap_embs

    torch.set_grad_enabled(True)
    return indices
