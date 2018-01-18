import tensorflow as tf
import numpy as np
import random
from collections import deque
from models.dqn import DQN
from torch import optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F


# Hyper Parameters:
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 32.  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 1000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
FINAL_EPSILON = 0
INITIAL_EPSILON = 0
# or alternative:
# FINAL_EPSILON = 0.0001  # final value of epsilon
# INITIAL_EPSILON = 0.01  # starting value of epsilon
UPDATE_TIME = 100
EXPLORE = 100000.  # frames over which to anneal epsilon


class RobotCNNDQN:

    # def __init__(self, actions=2, vocab_size=20000, max_len=120, embeddings=[]):
    def __init__(self, params):
        print("Creating a robot: CNN-DQN")
        # replay memory
        # TODO not use deque
        self.params = params
        self.replay_memory = deque()
        self.time_step = 0
        self.action = params["ACTIONS"]
        # self.max_len = max_len
        # self.num_classes = 5
        self.epsilon = INITIAL_EPSILON

        # self.vocab_size = vocab_size
        # self.max_len = max_len
        # self.embedding_size = 40
        self.qnetwork = DQN(params)

        if params["CUDA"]:
            self.qnetwork = self.qnetwork.cuda()
        # self.optimizer = optim.Adam(self.qnetwork.parameters(), 0.001)


    def initialise(self, params):
        self.qnetwork = DQN(params)
        # self.optimizer = optim.Adam(self.qnetwork.parameters(), 0.001)

    def train_qnetwork(self):
        # Step 1: obtain random minibatch from replay memory
        optimizer = optim.Adam(self.qnetwork.parameters(), 0.001)
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)

        batch_state, batch_action, batch_reward, batch_next_state, batch_terminal  = zip(*minibatch)
        batch_state = torch.cat(batch_state)
        batch_action = Variable(torch.LongTensor(list(batch_action)).unsqueeze(1))
        batch_reward = Variable(torch.FloatTensor(list(batch_reward)))
        #
        # batch_state.detach()
        # detached_batch_state = Variable(torch.FloatTensor(batch_state.detach().numpy()))

        detached_batch_state = Variable(torch.FloatTensor(batch_state.detach().cpu().data.numpy()))
        if self.params["CUDA"]:
            detached_batch_state = detached_batch_state.cuda()

        # batch_reward.detach()
        # batch_action.detach()

        if self.params["CUDA"]:
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
        # batch_reward.detach()

        batch_next_state = torch.cat(batch_next_state)
        current_q_values = self.qnetwork(detached_batch_state).gather(1, batch_action)
        # expected Q values are estimated from actions which gives maximum Q value
        max_next_q_values = self.qnetwork(batch_next_state).detach().max(1)[0]
        # print(max_next_q_values)
        expected_q_values = batch_reward + (GAMMA * max_next_q_values)

        if self.params["CUDA"]:
            # rand, rand2 = rand.cuda(), rand2.cuda()
            expected_q_values = expected_q_values.cuda()
        # print(rand2)
        # loss = F.l1_loss(current_q_values, expected_q_values)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # current_q_values.detach()
        # expected_q_values.detach()
        # loss.detach()
        print("AFTER STEP ")

    def update(self, observation, action, reward, observation2, terminal):
        self.replay_memory.append(
            (observation, action, reward, observation2, terminal))
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()
        global OBSERVE
        if self.time_step > OBSERVE:
            # Train the network
            self.train_qnetwork()

        self.time_step += 1

    def get_action(self, observation):
        # sentence, entropy = observation

        action = 0
        if random.random() <= self.epsilon:
            action = random.randrange(self.action)
        else:
            qvalue = self.qnetwork(observation)
            # print(qvalue.data[0])
            action = np.argmax(qvalue.data[0])
            # print("Returning action": action)
        # change episilon
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # action = torch.FloatTensor([action])
        return action
        # return 0

    # def process_sentence(self):
    #     seq_len = self.max_len
    #     vocab_size = self.vocab_size
    #     embedding_size = self.embedding_size
    #     filter_sizes = [3, 4, 5]
    #     num_filters = 128
    #
    #     self.sent = tf.placeholder(
    #         tf.int32, [None, seq_len], name="input_x")
    #     # dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    #     dropout_keep_prob = 0.5
    #     # Keeping track of l2 regularization loss (optional)
    #     l2_loss = tf.constant(0.0)
    #
    #     # embedding layer
    #     with tf.device('/cpu:0'), tf.name_scope("embedding"):
    #         # is able to train
    #         self.w = tf.Variable(tf.random_uniform(
    #             [self.vocab_size, embedding_size], -1.0, 1.0), trainable=False, name="W")
    #         self.embedded_chars = tf.nn.embedding_lookup(
    #             self.w, self.sent)
    #         self.embedded_chars_expanded = tf.expand_dims(
    #             self.embedded_chars, -1)
    #
    #     pooled_outputs = []
    #     for i, filter_size in enumerate(filter_sizes):
    #         with tf.name_scope("conv-maxpool-%s" % filter_size):
    #             # Convolution Layer
    #             filter_shape = [filter_size, embedding_size, 1, num_filters]
    #             W = tf.Variable(tf.truncated_normal(
    #                 filter_shape, stddev=0.1), name="W")
    #             b = tf.Variable(tf.constant(
    #                 0.1, shape=[num_filters]), name="b")
    #             conv = tf.nn.conv2d(
    #                 self.embedded_chars_expanded,
    #                 W,
    #                 strides=[1, 1, 1, 1],
    #                 padding="VALID",
    #                 name="conv")
    #             # Apply nonlinearity
    #             h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    #             # Maxpooling over the outputs
    #             pooled = tf.nn.max_pool(
    #                 h,
    #                 ksize=[1, seq_len - filter_size + 1, 1, 1],
    #                 strides=[1, 1, 1, 1],
    #                 padding='VALID',
    #                 name="pool")
    #             pooled_outputs.append(pooled)
    #     # Combine all the pooled features
    #     num_filters_total = num_filters * len(list(filter_sizes))
    #     h_pool = tf.concat(pooled_outputs, 3)
    #     h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    #     # Add dropout
    #     with tf.name_scope("dropout"):
    #         self.state_content = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    # TODO we just use entropy instead
    # def process_prediction(self):
    #     seq_len = self.max_len
    #     num_classes = self.num_classes
    #     # Placeholder for input
    #     self.predictions = tf.placeholder(
    #         tf.float32, [None, seq_len, num_classes], name="input_predictions")
    #     filter_sizes = [3]
    #     num_filters = 20
    #     self.predictions_expanded = tf.expand_dims(
    #         self.predictions, -1)
    #     dropout_keep_prob = 0.5
    #
    #     pooled_outputs = []
    #     for i, filter_size in enumerate(filter_sizes):
    #         with tf.name_scope("conv-maxpool-%s" % filter_size):
    #             # Convolution Layer
    #             filter_shape = [filter_size, num_classes, 1, num_filters]
    #             W = tf.Variable(tf.truncated_normal(
    #                 filter_shape, stddev=0.1), name="W")
    #             b = tf.Variable(tf.constant(
    #                 0.1, shape=[num_filters]), name="b")
    #             conv = tf.nn.conv2d(
    #                 self.predictions_expanded,
    #                 W,
    #                 strides=[1, 1, 1, 1],
    #                 padding="VALID",
    #                 name="conv")
    #             # Apply nonlinearity
    #             #h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    #             h = conv
    #             # averagepooling over the outputs
    #             pooled = tf.nn.avg_pool(
    #                 h,
    #                 ksize=[1, seq_len - filter_size + 1, 1, 1],
    #                 strides=[1, 1, 1, 1],
    #                 padding='VALID',
    #                 name="pool")
    #             pooled_outputs.append(pooled)
    #     # Combine all the pooled features
    #     num_filters_total = num_filters * len(list(filter_sizes))
    #     ph_pool = tf.concat(pooled_outputs, 3)
    #     ph_pool_flat = tf.reshape(ph_pool, [-1, num_filters_total])
    #     # Add dropout
    #     with tf.name_scope("dropout"):
    #         self.state_marginals = tf.nn.dropout(
    #             ph_pool_flat, dropout_keep_prob)

    # def update_embeddings(self, embeddings):
    #     # self.w_embeddings = embeddings
    #     # self.vocab_size = len(self.w_embeddings)
    #     # self.embedding_size = len(self.w_embeddings[0])
    #     print("Assigning new word embeddings")
    #     print("New size", self.vocab_size)
    #     self.sess.run(self.w.assign(self.w_embeddings))
    #     self.time_step = 0
    #     self.replay_memory = deque()
