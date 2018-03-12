from model import cosine_sim
from evaluation import encode_data
import torch.nn as nn
import random
import torch
from torch.autograd import Variable
import heapq
import scipy
import copy
import numpy as np
import sys

def select_random(r, model, train_loader):
    if r == 0:
        return random.sample(range(0, 30000), 1280)
    else:
        return random.sample(range(0, 30000), 128)

def select_margin(r, model, train_loader, primary="image"):
    if r == 0:
        return random.sample(range(0, 30000), 1280)
    else:
        model.val_start()
        img_embs, cap_embs = encode_data(model, train_loader)
        primary_embs, secondary_embs = (img_embs, cap_embs) if primary == "image" else (cap_embs, img_embs)

        scores = []

        for i in range(0, len(primary_embs), 128):
            batch_range = min(128, len(primary_embs) - i)
            primary_batch = primary_embs[i: i + batch_range]
            primary_batch = torch.FloatTensor(primary_batch)

            primary_secondary_distances = None
            if torch.cuda.is_available():
                primary_batch = primary_batch.cuda()

            for j in range(0, len(secondary_embs), 128):
                batch_range2 = min(128, len(secondary_embs) - j)
                secondary_batch = secondary_embs[j: j + batch_range2]

                secondary_batch = torch.FloatTensor(secondary_batch)
                if torch.cuda.is_available():
                    secondary_batch = secondary_batch.cuda()

                cosine_dist = primary_batch.mm(secondary_batch.t())

                if j == 0:
                    primary_secondary_distances = cosine_dist
                else:
                    primary_secondary_distances = torch.cat((primary_secondary_distances, cosine_dist), 1)

            distances_top2 = torch.abs(torch.topk(primary_secondary_distances, 2, 1, largest=False)[0])
            margin = torch.abs(distances_top2[:, 0] - distances_top2[:, 1])

            scores.extend(margin.cpu().numpy())
            print 'Selection: {:2.4}%\r'.format((float(i) / float(len(primary_embs))) * 100),

        best_n_indices = [n[0] for n in heapq.nsmallest(128, enumerate(scores), key=lambda x: x[1])]
        return best_n_indices

def select_captionSimilarity(r, model, train_loader, primary="image"):
    if r == 0:
        return random.sample(range(0, 30000), 1280)
    else:    
        model.val_start()
        img_embs, cap_embs = encode_data(model, train_loader)
        primary_embs, secondary_embs = (img_embs, cap_embs) if primary == "image" else (cap_embs, img_embs)

        scores = []

        for i in range(0, len(cap_embs), 5):
            batch_range = min(5, len(cap_embs) - i)
            primary_batch = cap_embs[i: i + batch_range]
            primary_batch = Variable(torch.FloatTensor(primary_batch))
            primary_batch_sum = get_avg_distance(primary_batch)
            # print(primary_batch_sum)
            scores.append(primary_batch_sum.data.cpu().numpy())

        #Best results        
        best_n_indices = [n[0] for n in heapq.nsmallest(128, enumerate(scores), key=lambda x: x[1])]    
        print(best_n_indices)
        #worst results
        # best_n_indices = [n[0] for n in heapq.nlargest(128, enumerate(scores), key=lambda x: x[1])]

        return best_n_indices

def get_avg_distance(caption_set):
    average_cap = 0.0
    total_cap_vector = torch.sum(caption_set, dim=0)
    average_cap_vector = torch.div(total_cap_vector,len(caption_set))
    average_cap_vector.unsqueeze(0)
    cosine_dist = average_cap_vector.unsqueeze(0).mm(caption_set.t())
    average_distance = torch.sum(cosine_dist)
    return average_distance

def select_uncertainty(r, model, train_loader, primary="image"):
    if r == 0:
        return random.sample(range(0, 30000), 1280)
    else:
        model.val_start()
        img_embs, cap_embs = encode_data(model, train_loader)
        primary_embs, secondary_embs = (img_embs, cap_embs) if primary == "image" else (cap_embs, img_embs)

        scores = []
        
        for i in range(0, len(primary_embs), 128):
            batch_range = min(128, len(primary_embs) - i)
            primary_batch = primary_embs[i: i + batch_range]
            primary_batch = torch.FloatTensor(primary_batch)

            image_scores = torch.zeros(batch_range)

            primary_secondary_distances = None
            if torch.cuda.is_available():
                primary_batch = primary_batch.cuda()
                image_scores = image_scores.cuda()

            for j in range(0, len(secondary_embs), 128):
                batch_range2 = min(128, len(secondary_embs) - j)
                secondary_batch = secondary_embs[j: j + batch_range2]

                secondary_batch = torch.FloatTensor(secondary_batch)
                if torch.cuda.is_available():
                    secondary_batch = secondary_batch.cuda()

                cosine_dist = primary_batch.mm(secondary_batch.t())

                if j == 0:
                    primary_secondary_distances = cosine_dist
                else:
                    primary_secondary_distances = torch.cat((primary_secondary_distances, cosine_dist), 1)

            # print(primary_secondary_distances)
            distances_top10_index = torch.abs(torch.topk(primary_secondary_distances, 10, 1, largest=False)[1])

            for row in distances_top10_index.cpu().numpy():
                std = np.array([])

                for caption_index in row:
                    std = np.concatenate((std, secondary_embs[caption_index]))
                scores.append(np.std(std))
            print 'Selection: {:2.4}%\r'.format((float(i) / float(len(primary_embs))) * 100),

        best_n_indices = [n[0] for n in heapq.nlargest(128, enumerate(scores), key=lambda x: x[1])]
        return best_n_indices


def select_hybrid(r, model, train_loader):
    if r == 0:
        return random.sample(range(0, 30000), 1280)
    elif r%2 == 0:
        return select_uncertainty(r, model, train_loader)
    else:
        return select_margin(r, model, train_loader)

def select_all(r, model, train_loader):
    if r == 0:   
        size = ((len(train_loader)*128)-128)
        # print(size)
        return range(0, size) 
    else: 
        sys.exit("Done!")