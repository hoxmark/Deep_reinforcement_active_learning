import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import numpy as np

from utils import pairwise_distances, batchify
from config import opt, data, loaders

from tensorboardX import SummaryWriter

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.input_size = 784
        # TODO params
        self.hidden_size = 548
        self.output_size = 10

        ##FIRST
        # self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # self.fc3 = nn.Linear(self.hidden_size, self.output_size)

        ##SEC
        # self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # self.bc1 = nn.BatchNorm1d(self.hidden_size)        
        # self.fc2 = nn.Linear(self.hidden_size, 252)
        # self.bc2 = nn.BatchNorm1d(252)        
        # self.fc3 = nn.Linear(252, self.output_size)
        
        ##THIRD
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        if opt.cuda:
            self.cuda()

        self.reset()

        self.writer = SummaryWriter(comment='mnist_embedding_training')

    def reset(self):
        # pass
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        if opt.cuda:
            self.cuda()

        # torch.nn.init.xavier_normal_(self.fc3.weight)

    # def forward(self, inp):
    #     x = Variable(inp)
    #     if opt.cuda:
    #         x = x.cuda()

    #     h = self.fc1(x)
    #     h = self.bc1(h)
    #     h = F.relu(h)
    #     h = F.dropout(h, p=0.5, training=self.training)
        
    #     h = self.fc2(h)
    #     h = self.bc2(h)
    #     h = F.relu(h)
    #     h = F.dropout(h, p=0.2, training=self.training)
        
    #     h = self.fc3(h)
    #     # h = self.relu(h)
    #     out = F.log_softmax(h)
    #     return out
    
    # def forward(self, inp):
        # x = Variable(inp)

        # if opt.cuda:
        #     inp = inp.cuda()
        # output = self.fc1(inp)
        # output = self.relu(output)
        # output = self.fc3(output)
        # return output

    def forward(self, x):        
        x = Variable(x)

        if opt.cuda:
            x = x.cuda()   

        # print(x.size())
        x = x.view(len(x),1, 28, 28)
        # print(x.size())       
         
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


    def train_model(self, data, epochs):
        
        # import keyword
        # import torch
        # meta = []
        # while len(meta)<100:
        #     meta = meta+keyword.kwlist # get some strings
        # meta = meta[:100]
        # for i, v in enumerate(meta):
        #     meta[i] = v+str(i)
        # label_img = torch.rand(100, 3, 10, 32)
        # for i in range(100):
        #     label_img[i]*=i/100.0
        # self.writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
        # # self.writer.add_embedding(torch.randn(100, 5), label_img=label_img)
        # # self.writer.add_embedding(torch.randn(100, 5), metadata=meta)
        # self.writer.close()
        # quit()

        # optimizer = optim.Adadelta(self.parameters(), 0.1)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.train()
        size = len(data[0])
        

        for e in range(epochs):
            avg_loss = 0
            corrects = 0
            for i, (features, targets) in enumerate(batchify(data)):
                features = Variable(torch.FloatTensor(features))
                targets = Variable(torch.LongTensor(targets))
                if opt.cuda:
                    features, targets = features.cuda(), targets.cuda()

                output = self.forward(features)
                # output = output.cuda()

                x = features.view(len(output), 1, 28, 28)
                y = targets.view(len(output))
                
                print(y.size())

                n_iter = (e*128) + i
                optimizer.zero_grad()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                avg_loss += loss.data.item()
                corrects += (torch.max(output, 1)
                             [1].view(targets.size()).data == targets.data).sum()
                #LOGGING
                # self.writer.add_scalar('loss', loss.data[0], n_iter)
                if i % 5 == 0:
                    # print("loss_value:{}".format(loss.data[0]))
                    #we need 3 dimension for tensor to visualize it!
                    a = torch.ones(len(output), 1)
                    a = a.cuda()
                    out = torch.cat((output.data, a), 1)                    
            
            
        
            avg_loss = avg_loss / opt.batch_size
            accuracy = 100.0 * corrects / size

        print("DONE")
        # self.writer.close()
        # quit()
        


    def predict_prob(self, inp):
        with torch.no_grad():
            output = self.forward(inp)
            # print(output)

            output = torch.nn.functional.softmax(output, dim=1)
            # print(output)
            # quit()
            return output

    def validate(self, data):
        corrects, avg_loss = 0, 0
        with torch.no_grad():
            for i, (features, targets) in enumerate(batchify(data)):
                features = Variable(torch.FloatTensor(features))
                targets = Variable(torch.LongTensor(targets))

                if opt.cuda:
                    features = features.cuda()
                    targets = targets.cuda()

                logit = self.forward(features)
                # loss = torch.nn.functional.nll_loss(logit, targets, size_average=False)
                loss = torch.nn.functional.cross_entropy(logit, targets, size_average=False)
                avg_loss += loss.data.item()
                corrects += (torch.max(logit, 1)[1].view(targets.size()).data == targets.data).sum()

            size = len(data[0])
            avg_loss = avg_loss / size
            accuracy = 100.0 * float(corrects) / float(size)

            metrics = {
                'accuracy': accuracy,
                'avg_loss': avg_loss,
                'performance': accuracy
            }

            return metrics

    def performance_validate(self, data):
        return self.validate(data)

    def get_state(self, index):
        state = data["all_predictions"][index].view(1, -1)
        state = state.sort()[0]
        if opt.cuda:
            state = state.cuda()
        return state

    def encode_episode_data(self):
        images = []
        # for i, (features, targets) in enumerate(loaders["train_loader"]):
        for i, (features, targets) in enumerate(batchify(data["train_deleted"])):
            features = Variable(torch.FloatTensor(features))
            targets = Variable(torch.LongTensor(targets))
            preds = self.predict_prob(features)
            
            images.append(preds)
        
        images = torch.cat(images, dim=0)
        data["all_predictions"] = images


    def find_avg_pred_entropy(self, model):
        images = []
        tot = 0
        length = len(data["all_predictions"])

        for i, pred in enumerate(data["all_predictions"]):
            tot += entropy(pred)

        avg = tot/length
        return avg

    def query(self, index):
        current_state = data["all_predictions"][index].view(1, -1)
        all_states = data["all_predictions"]
        current_all_dist = pairwise_distances(current_state, all_states)
        similar_indices = torch.topk(current_all_dist, opt.selection_radius, 1, largest=False)[1]
        # self.visualize(similar_indices)
        similar_indices = similar_indices.data[0].cpu().numpy()
        

        for idx in similar_indices:
            self.add_index(idx)
        return similar_indices

    def add_index(self, index):
        image = data["train_deleted"][0][index]
        caption = data["train_deleted"][1][index]
        data["active"][0].append(image)
        data["active"][1].append(caption)

    # def visualize(self, inp_indices):
    #     print(inp_indices)
    #     # index = inp_indices[0]
    #     chosen = data["all_predictions"][inp_indices]
    #     all_data = data["all_predictions"]
    #     labels = torch.zeros(len(data["all_predictions"]))
    #     print(all_data.size())
    #     print(labels.size())
        
    #     self.writer.add_embedding(all_data, global_step=0)        
    #     # self.writer.add_embedding(chosen, global_step=1)        
    #     # self.writer.add_embedding(all_data, global_step=1)        
    #     self.writer.close()
    #     quit()  

                
    def visualize(self, added_indices, data):
        
        all_data = torch.LongTensor(data[0])
        # chosen = torch.FloatTensor(chosen[0])
        # print(all_data)

        # added_indices = [0,1,2,3,4,5,6]
        # print(added_indices)
        labels = torch.zeros(len(data[0]))
        labels[added_indices] = 1
        print(labels.size())
        print(all_data.size())
        self.writer.add_embedding(all_data.data, metadata=labels.data, global_step=0)

        self.writer.close()
        quit()  


        # self.writer.add_embedding(chosen.data, global_step=1)

        # for i, (features, targets) in enumerate(batchify(data)):
        #     features = Variable(torch.FloatTensor(features))
        #     targets = Variable(torch.FloatTensor(targets))
        #     print(features.size())
        #     # print()
        
        #     self.writer.add_embedding(features.data, metadata=targets.data, global_step=i)
        #     # self.writer.add_embedding(features.data,  global_step=i)
        


        # # quit()
        # # label_batch = Variable(sample[1], requires_grad=False).long() 
        # print(inp_indices)
        # print(all_data.size())
        # # index = inp_indices[0]
        # chosen = data["train_deleted"][0][inp_indices]
        # # labels = torch.zeros(len(data["train_deleted"]))
        # print(all_data.size())
        # print(labels.size())
        
        # self.writer.add_embedding(chosen, global_step=0)        
        # # self.writer.add_embedding(chosen, global_step=1)        
        # # self.writer.add_embedding(all_data, global_step=1)        