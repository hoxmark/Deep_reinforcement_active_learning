from model import cosine_sim
from evaluation import encode_data

import torch
import heapq
import scipy
import copy

def select_margin(model, train_loader):
    img_embs, cap_embs = encode_data(model, train_loader)
    scores = []

    for i in range(0, len(img_embs), 128):
        batch_range = min(128, len(img_embs) - i)
        img_batch = img_embs[i: i + batch_range]
        img_batch = torch.FloatTensor(img_batch)

        image_scores = torch.zeros(batch_range)

        if torch.cuda.is_available():
            img_batch = img_batch.cuda()
            image_scores = image_scores.cuda()

        for j in range(0, len(cap_embs), 128):
            batch_range2 = min(128, len(cap_embs) - i)
            cap_batch = cap_embs[i: i + batch_range2]

            cap_batch = torch.FloatTensor(cap_batch)
            if torch.cuda.is_available():
                cap_batch = cap_batch.cuda()

            distances = img_batch.mm(cap_batch.t())

            distances_top2 = torch.abs(torch.topk(distances, 2, 1, largest=False)[0])
            margin = torch.abs(distances_top2[:, 0] - distances_top2[:, 1])
            image_scores += margin

        image_scores = torch.div(image_scores, len(img_embs) / batch_range)
        scores.extend(image_scores.cpu().numpy())

        print 'Selection: {:2.4}%\r'.format((float(i) / float(len(img_embs))) * 100),

    best_n_indexes = [n[0] for n in heapq.nsmallest(128, enumerate(scores), key=lambda x: x[1])]
    return best_n_indexes
