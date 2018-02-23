from model import cosine_sim
import torch
import heapq


def select_margin(model, train_loader):
    scores = []
    for i, train_data in enumerate(train_loader):

        images, captions, lengths = train_data[0], train_data[1], train_data[2]
        image_scores = torch.zeros(128)

        if torch.cuda.is_available():
            image_scores = image_scores.cuda()
        for j, train_data2 in enumerate(train_loader):
            captions2, lengths2 = train_data2[1], train_data2[2]


            img_emb, cap_emb = model.forward_emb(images, captions2, lengths2)

            distances = cosine_sim(img_emb, cap_emb)
            distances_top2 = torch.abs(torch.topk(distances, 2, 1, largest=False)[0])
            margin = torch.abs(distances_top2[:, 0] - distances_top2[:, 1])
            # print(margin)
            # print(margin.data)
            image_scores += margin.data

        print("{} of {}\r".format(i, len(train_loader)))
        image_scores = torch.div(image_scores, len(train_loader) / 128)
        scores.extend(image_scores.cpu().numpy())


    best_n_indexes = [n[0] for n in heapq.nsmallest(128, enumerate(scores), key=lambda x: x[1])]
    print(best_n_indexes)

    return best_n_indexes
