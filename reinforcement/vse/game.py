import numpy as np
import sys
import time
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from config import opt, data, loaders
from data.utils import timer, average_vector, get_distance, pairwise_distances
# from data.evaluation import encode_data, i2t, t2i
from data.dataset import get_active_loader
from scipy.stats import entropy


class Game:
    def reboot(self, model):
        """resets the Game Object, to make it ready for the next episode """

        loaders["active_loader"] = get_active_loader(opt.batch_size)
        data_len = loaders["train_loader"].dataset.length-opt.init_samples
        # self.order = random.sample(list(range(0, data_len // 5)), data_len // 5)
        self.order = random.sample(list(range(0, data_len)), data_len)
        self.budget = opt.budget
        self.queried_times = 0
        self.current_state = 0
        self.entropy = 0
        self.init_train_k_random(model, opt.init_samples)
        self.avg_entropy_in_train_loader = self.get_avg_entropy_in_train_loader(loaders["train_loader"])
        # if opt.embedding != 'static':
            # self.encode_episode_data(model, loaders["train_loader"])
        self.performance =  model.validate(loaders["val_loader"])

    def get_avg_entropy_in_train_loader(self,loader): 
        images = []
        total = 0
        
        # np.set_printoptions(threshold=np.inf)        
        length = len(loader.dataset)
        for i, train_data in enumerate(loader.dataset): 
            total += entropy(train_data[0])

        print(length)
        return total/length


    def encode_episode_data(self, model, loader):
        img_embs, cap_embs = timer(model.encode_data, (loader,))
        captions = torch.FloatTensor(cap_embs)
        images = []

        # TODO dynamic im_div
        for i in range(0, len(img_embs), 5):
            images.append(img_embs[i])
        images = torch.FloatTensor(images)

        image_caption_distances = pairwise_distances(images, captions)
        image_caption_distances_topk = torch.topk(image_caption_distances, opt.topk, 1, largest=False)[0]

        data["images_embed_all"] = images
        data["captions_embed_all"] = captions
        data["image_caption_distances_topk"] = image_caption_distances_topk
        # data["img_embs_avg"] = average_vector(data["images_embed_all"])
        # data["cap_embs_avg"] = average_vector(data["captions_embed_all"])

    def get_state(self, model):
        current_idx = self.order[self.current_state]
        # observation = self.construct_distance_state(current_idx)
        observation = self.construct_entropy_state(model, current_idx)
        self.current_state += 1        
        returnOb = Variable(torch.FloatTensor(observation))
        returnOb = returnOb.cuda()        
        return returnOb

    def construct_distance_state(self, index):
        # Distances to topk closest captions
        image_topk = data["image_caption_distances_topk"][index].view(1, -1)
        state = image_topk

        # Distances to topk closest images
        if opt.topk_image > 0:
            current_image = data["images_embed_all"][index].view(1 ,-1)
            all_images = data["images_embed_all"]
            image_image_dist = pairwise_distances(current_image, all_images)
            image_image_dist_topk = torch.topk(image_image_dist, opt.topk_image, 1, largest=False)[0]

            state = torch.cat((state, image_image_dist_topk), 1)

        # Distance from average image vector
        if opt.image_distance:
            current_image = data["images_embed_all"][index].view(1 ,-1)
            img_distance = get_distance(current_image, data["img_embs_avg"].view(1, -1))
            image_dist_tensor = torch.FloatTensor([img_distance]).view(1, -1)
            state = torch.cat((state, image_dist_tensor), 1)

        observation = torch.autograd.Variable(state)
        if opt.cuda:
            observation = observation.cuda()
        return observation

    def construct_entropy_state(self, model, index):
        current_image = loaders["train_loader"].dataset[index][0]
     
        preds = model.predict_prob([current_image])

        return preds

    def construct_all_predictions(self, model):
        # all_predictions = torch.FloatTensor()
        all_predictions = None
        # if opt.cuda:
            # all_predictions = all_predictions.cuda()
        images = []
        
        for i, train_data in enumerate(loaders["train_loader"].dataset): 
            images.append(train_data[0])
        preds = model.predict_prob(images)
        assert(len(preds[0])==10)
        # preds = nn.functional.softmax(preds, dim=1)
        # if i == 0:
        #     all_predictions = preds
        # else:
        #     all_predictions = torch.cat((all_predictions, preds), dim=0)
        del images
        
        #TODO make tensor or not? 
        # data["all_predictions"] = Variable(torch.FloatTensor(preds), requires_grad=False, volatile=True)
        data["all_predictions"] = preds
    
    def find_avg_pred_entropy(self, model):
        images = []
        tot = 0
        length = len(data["all_predictions"])

        for i, pred in enumerate(data["all_predictions"]):
            tot += entropy(pred)
        
        avg = tot/length        
        return avg

    # def find_entropy(self, inp)

    def feedback(self, action, model):
        reward = 0.
        is_terminal = False
        if action == 1:
            # timer(self.query, (model,))
            is_terminal = self.query(model)
            if is_terminal: 
                return None, None, True
            new_performance = self.get_performance(model)
            # print("old performance: {}".format(self.performance))
            # print("new performance: {}".format(new_performance))
            
            # reward = self.performance - new_performance
            # reward = new_performance - self.performance
            # reward -= -0.4
            # print("reward: {}".format(reward))
            # reward = entropy - 0.6
            # self.entropy = self.find_entropy(model)
            
            #find entropy from img entropy
            # avg_active = self.get_avg_entropy_in_train_loader(loaders["active_loader"])
            # avg_train = self.get_avg_entropy_in_train_loader(loaders["train_loader"])
            # current = self.order[self.current_state]            
            # current_state = data["all_predictions"][current]
            # e = entropy(current_state)            
            # # reward = e - float(opt.reward)
            # reward = self.entropy - self.avg_entropy_in_train_loade
            # # self.entropy = e 
            # print("active_avg       {}        train_avg   {}".format(avg_active, avg_train))
            # print("entropy:         {}        reward:     {}".format(self.entropy, reward) )
            # print("current_state:   {}        e from c_s: {}".format(current_state, e) )

            #find entropy of predictions
            avg_pred_entropy = self.find_avg_pred_entropy(model)
            current = self.order[self.current_state]            
            current_state = data["all_predictions"][current]
            e = entropy(current_state)
            reward = e - avg_pred_entropy - float(opt.reward)
            
            # print("entropy: {}   -  reward:{}".format(e, reward) )
            # print("entropy:         {}        reward:     {}".format(self.entropy, (self.entropy - self.avg_entropy_in_train_loader) ))
        
            if opt.reward_clip:
                reward = np.tanh(reward / 100)

            # print("reward: {} after tanh".format(reward))
            self.performance = new_performance
        else:
            reward = 0.
        # TODO fix this
        if self.queried_times >= self.budget or self.current_state >= len(self.order):
            # Return terminal
            return None, None, True

        print("> State {:2} Action {:2} - reward {:.4f} - accuracy {:.4f}".format(
            self.current_state, action, reward, self.performance))
        # next_observation = timer(self.get_state, (model,))
        next_observation = self.get_state(model)
        return reward, next_observation, is_terminal

    def query(self, model):
        self.construct_all_predictions(model)        
        if (len(self.order) == self.current_state):            
            return True
        current = self.order[self.current_state]
        image = loaders["train_loader"].dataset[current][0]
        self.entropy = entropy(loaders["train_loader"].dataset[current][0][0])

        caption = loaders["train_loader"].dataset[current][1]
        
        # There are 5 captions for every image
        loaders["active_loader"].dataset.add_single(image, caption)

        self.queried_times += 1
        
        return False
    # def query(self, model):
    #     self.construct_all_predictions(model)        
    #     if (len(self.order) == self.current_state):            
    #         return True
    #     current = self.order[self.current_state]
    #     # construct_state = self.construct_entropy_state
    #     #
    #     # current_state = construct_state(model, current)
    #     # all_states = torch.cat([construct_state(model, index) for index in range(len(self.order))])
        
    #     current_state = data["all_predictions"][current]
    #     all_states = data["all_predictions"]
    #     current_state = torch.from_numpy(current_state).view(1,-1)
    #     all_states = torch.from_numpy(all_states)
    #     current_all_dist = pairwise_distances(current_state, all_states)
    #     similar_indices = torch.topk(current_all_dist, opt.selection_radius, 1, largest=False)[1]

    #     for index in similar_indices.cpu().numpy():
    #         image = loaders["train_loader"].dataset[index][0]

    #         self.entropy = entropy(loaders["train_loader"].dataset[index][0][0])
    #         caption = loaders["train_loader"].dataset[index][1]
    #         # There are 5 captions for every image
    #         loaders["active_loader"].dataset.add_single(image[0], caption[0])

    #         # Only count images as an actual request.
    #         # Reuslt is that we have 5 times as many training points as requests.
    #         self.queried_times += 1
        
    #     return False
        # for index in similar_indices[0]:
        #     image = loaders["train_loader"].dataset[5 * index][0]
        #     # There are 5 captions for every image
        #     for cap in range(5):
        #         caption = loaders["train_loader"].dataset[5 * index + cap][1]
        #         loaders["active_loader"].dataset.add_single(image, caption)
        #     # Only count images as an actual request.
        #     # Reuslt is that we have 5 times as many training points as requests.
        #     self.queried_times += 1

    def init_train_k_random(self, model, num_of_init_samples):        
        for i in range(0, num_of_init_samples):
            current = self.order[(-1*(i + 1))]
            image = loaders["train_loader"].dataset[current][0]
            caption = loaders["train_loader"].dataset[current][1]
            loaders["active_loader"].dataset.add_single(image, caption)
            
        # timer(model.train_model, (loaders["active_loader"], opt.num_epochs))
        model.train_model(loaders["active_loader"], opt.num_epochs)

        # print("Validation after training on random data: {}".format(model.validate(loaders["val_loader"])))
        
    def get_performance(self, model):
        # timer(self.train_model, (model, loaders["active_loader"]))
        # timer(model.train_model, (loaders["active_loader"], opt.num_epochs))
        model.train_model(loaders["active_loader"], opt.num_epochs)
        # timer(model.train_model, (loaders["active_loader"], opt.num_epochs))        
        metrics = model.validate(loaders["val_loader"])
        performance = metrics
        print("performance: {}".format(performance))
        # if (self.queried_times % 20 == 0):
            # if opt.embedding != 'static':
        # self.encode_episode_data(model, loaders["train_loader"])
        return performance


    # def train(self, model, train_loader, epochs=opt.num_epochs):

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR
           decayed by 10 every 30 epochs"""
        lr = opt.learning_rate_vse * (0.1 ** (epoch // opt.lr_update))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
