    # def encode_episode_data(self, model, loader):
    #     img_embs, cap_embs = timer(model.encode_data, (loader,))
    #     captions = torch.FloatTensor(cap_embs)
    #     images = []
    #
    #     # TODO dynamic im_div
    #     for i in range(0, len(img_embs), 5):
    #         images.append(img_embs[i])
    #     images = torch.FloatTensor(images)
    #
    #     image_caption_distances = pairwise_distances(images, captions)
    #     image_caption_distances_topk = torch.topk(image_caption_distances, opt.topk, 1, largest=False)[0]
    #
    #     data["images_embed_all"] = images
    #     data["captions_embed_all"] = captions
    #     data["image_caption_distances_topk"] = image_caption_distances_topk
    #     # data["img_embs_avg"] = average_vector(data["images_embed_all"])
    #     # data["cap_embs_avg"] = average_vector(data["captions_embed_all"])


    # def construct_distance_state(self, index):
    #     # Distances to topk closest captions
    #     image_topk = data["image_caption_distances_topk"][index].view(1, -1)
    #     state = image_topk
    #
    #     # Distances to topk closest images
    #     if opt.topk_image > 0:
    #         current_image = data["images_embed_all"][index].view(1 ,-1)
    #         all_images = data["images_embed_all"]
    #         image_image_dist = pairwise_distances(current_image, all_images)
    #         image_image_dist_topk = torch.topk(image_image_dist, opt.topk_image, 1, largest=False)[0]
    #
    #         state = torch.cat((state, image_image_dist_topk), 1)
    #
    #     # Distance from average image vector
    #     if opt.image_distance:
    #         current_image = data["images_embed_all"][index].view(1 ,-1)
    #         img_distance = get_distance(current_image, data["img_embs_avg"].view(1, -1))
    #         image_dist_tensor = torch.FloatTensor([img_distance]).view(1, -1)
    #         state = torch.cat((state, image_dist_tensor), 1)
    #
    #     observation = torch.autograd.Variable(state)
    #     if opt.cuda:
    #         observation = observation.cuda()
    #     return observation



# def query():
#     current_state = data["all_predictions"][current]
#     all_states = data["all_predictions"]
#     current_state = torch.from_numpy(current_state).view(1,-1)
#     all_states = torch.from_numpy(all_states)
#     current_all_dist = pairwise_distances(current_state, all_states)
#     similar_indices = torch.topk(current_all_dist, opt.selection_radius, 1, largest=False)[1]

    # for index in similar_indices[0]:
    #     image = loaders["train_loader"].dataset[5 * index][0]
    #     # There are 5 captions for every image
    #     for cap in range(5):
    #         caption = loaders["train_loader"].dataset[5 * index + cap][1]
    #         loaders["active_loader"].dataset.add_single(image, caption)
    #     # Only count images as an actual request.
    #     # Reuslt is that we have 5 times as many training points as requests.
    #     self.queried_times += 1



    #         current = self.order[self.current_state]
    #         image = loaders["train_loader"].dataset[current][0]
    #         self.entropy = entropy(loaders["train_loader"].dataset[current][0][0])
    #
    #         caption = loaders["train_loader"].dataset[current][1]
    #
    #         # There are 5 captions for every image
    #         loaders["active_loader"].dataset.add_single(image, caption)
    #
    #         self.queried_times += 1
    #
    #         return False
        # def query(self, model):
        #     self.construct_all_predictions(model)
        #     if (len(self.order) == self.current_state):
        #         return True
        #     current = self.order[self.current_state]
        #     # construct_state = self.construct_entropy_state
        #     #
        #     # current_state = construct_state(model, current)
        #     # all_states = torch.cat([construct_state(model, index) for index in range(len(self.order))])



        #     for index in similar_indices.cpu().numpy():
        #         image = loaders["train_loader"].dataset[index][0]

        #         self.entropy = entropy(loaders["train_loader"].dataset[index][0][0])
        #         caption = loaders["train_loader"].dataset[index][1]
        #         # There are 5 captions for every image
        #         loaders["active_loader"].dataset.add_single(image[0], caption[0])

        #         # Only count images as an actual request.
        #         # Reuslt is that we have 5 times as many training points as requests.
        #         self.queried_times += 1

#
# def adjust_learning_rate(self, optimizer, epoch):
#     """Sets the learning rate to the initial LR
#        decayed by 10 every 30 epochs"""
#     lr = opt.learning_rate_vse * (0.1 ** (epoch // opt.lr_update))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
