from model import cosine_sim
from evaluation import encode_data
import random
import torch
import heapq
import scipy
import copy
import numpy as np

def select_random(model, train_loader):
    return random.sample(range(0, 30000), 128)

def select_margin(model, train_loader):
    img_embs, cap_embs = encode_data(model, train_loader)
    scores = []

    for i in range(0, len(img_embs), 128):
        batch_range = min(128, len(img_embs) - i)
        img_batch = img_embs[i: i + batch_range]
        img_batch = torch.FloatTensor(img_batch)

        img_cap_distances = None
        if torch.cuda.is_available():
            img_batch = img_batch.cuda()

        for j in range(0, len(cap_embs), 128):
            batch_range2 = min(128, len(cap_embs) - j)
            cap_batch = cap_embs[j: j + batch_range2]

            cap_batch = torch.FloatTensor(cap_batch)
            if torch.cuda.is_available():
                cap_batch = cap_batch.cuda()

            cosine_dist = img_batch.mm(cap_batch.t())

            if j == 0:
                img_cap_distances = cosine_dist
            else:
                img_cap_distances = torch.cat((img_cap_distances, cosine_dist), 1)

        distances_top2 = torch.abs(torch.topk(img_cap_distances, 2, 1, largest=False)[0])
        margin = torch.abs(distances_top2[:, 0] - distances_top2[:, 1])

        scores.extend(margin.cpu().numpy())
        print 'Selection: {:2.4}%\r'.format((float(i) / float(len(img_embs))) * 100),

    best_n_indices = [n[0] for n in heapq.nsmallest(128, enumerate(scores), key=lambda x: x[1])]
    return best_n_indices

def select_uncertainty(model, train_loader):
    img_embs, cap_embs = encode_data(model, train_loader)
    scores = []

    for i in range(0, len(img_embs), 128):
        batch_range = min(128, len(img_embs) - i)
        img_batch = img_embs[i: i + batch_range]
        img_batch = torch.FloatTensor(img_batch)

        image_scores = torch.zeros(batch_range)

        img_cap_distances = None
        if torch.cuda.is_available():
            img_batch = img_batch.cuda()
            image_scores = image_scores.cuda()

        for j in range(0, len(cap_embs), 128):
            batch_range2 = min(128, len(cap_embs) - j)
            cap_batch = cap_embs[j: j + batch_range2]

            cap_batch = torch.FloatTensor(cap_batch)
            if torch.cuda.is_available():
                cap_batch = cap_batch.cuda()

            cosine_dist = img_batch.mm(cap_batch.t())
            # print(cosine_dist)
            # closest_10 = torch.topk(cosine_dist, 10, 1, largest=False)[0]

            if j == 0:
                img_cap_distances = cosine_dist
            else:
                img_cap_distances = torch.cat((img_cap_distances, cosine_dist), 1)

        # print(img_cap_distances)
        distances_top10_index = torch.abs(torch.topk(img_cap_distances, 10, 1, largest=False)[1])

        for row in distances_top10_index.cpu().numpy():
            std = np.array([])

            for caption_index in row:
                std = np.concatenate((std, cap_embs[caption_index]))
            scores.append(np.std(std))
        print 'Selection: {:2.4}%\r'.format((float(i) / float(len(img_embs))) * 100),

    best_n_indices = [n[0] for n in heapq.nlargest(128, enumerate(scores), key=lambda x: x[1])]
    return best_n_indices
