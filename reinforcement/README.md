# Active reinforcement learning for visual semantic embedding

## Download datasets and w2v
We use the Flickr8k dataset. Splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The precomputed image features are from [here](https://github.com/ryankiros/visual-semantic-embedding/) and [here](https://github.com/ivendrov/order-embedding).

To download the precomputed image-features and the vocabulary:
```
wget http://www.cs.toronto.edu/~faghri/vsepp/vocab.tar
wget http://www.cs.toronto.edu/~faghri/vsepp/data.tar
```

Download w2v
```
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```

## Running
The first argument has to be which dataset to run the experiment on. This is so we can split on that and conditionally add parameters to the parser depending on which dataset is being used. After the dataset argument, the following arguments are allowed, depending on dataset

### Flickr8k
```
--hidden_size HIDDEN_SIZE               Size of hidden layer in deep RL
--episodes EPISODES                     Number of episodes
--learning_rate_rl LEARNING_RATE_RL     Learning rate
--margin MARGIN                         Rank loss margin.
--num_epochs NUM_EPOCHS                 Number of reward calculation epochs.
--full_epochs FULL_EPOCHS               Number of training epochs.
--init_samples INIT_SAMPLES             Number of random inital training data
--batch_size BATCH_SIZE                 Size of a training mini-batch.
--budget BUDGET                         Labeling budget
--selection_radius SELECTION_RADIUS     Selection radius
--reward_threshold REWARD_THRESHOLD     Reward threshold
--scorefn SCOREFN                       Score FN for traditional active learning
--w2v                                   Use w2v embeddings
--embed_size EMBED_SIZE                 Dimensionality of the joint embedding.
--word_dim WORD_DIM                     Dimensionality of the word embedding.
--num_layers NUM_LAYERS                 Number of GRU layers.
--grad_clip GRAD_CLIP                   Gradient clipping threshold.
--crop_size CROP_SIZE                   Size of an image crop as the CNN input.
--learning_rate_vse LEARNING_RATE_VSE   Initial learning rate.
--lr_update LR_UPDATE                   Number of epochs to update the learning rate.
--workers WORKERS                       Number of data loader workers.
--log_step LOG_STEP                     Number of steps to print and record the log.
--val_step VAL_STEP                     Number of steps to run validation.
--img_dim IMG_DIM                       Dimensionality of the image embedding.
--cnn_type CNN_TYPE                     The CNN used for image encoder(e.g. vgg19, resnet152)
--topk TOPK                             Topk similarity to use for state
--topk_image TOPK_IMAGE                 Topk similarity images to use for state
--data_name DATA_NAME                   {coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k
--measure MEASURE                       Similarity measure used (cosine|order)
--intra_caption                         Include closest captions intra distance in state
--max_violation                         Use max instead of sum in the rank loss.
--image_distance                        Include image distance in the state
--use_abs                               Take the absolute value of embedding vectors.
--no_imgnorm                            Do not normalize the image embeddings.
--finetune                              Fine-tune the image encoder.
--use_restval                           Use the restval data for training on MSCOCO.
```

### MR
```
--hidden_size HIDDEN_SIZE           Size of hidden layer in deep RL
--episodes EPISODES                 Number of episodes
--learning_rate_rl LEARNING_RATE_RL learning rate
--margin MARGIN                     Rank loss margin.
--num_epochs NUM_EPOCHS             Number of training epochs.
--full_epochs FULL_EPOCHS           Number of training epochs.
--init_samples INIT_SAMPLES         Number of random inital training data
--batch_size BATCH_SIZE             Size of a training mini-batch.
--budget BUDGET                     Our labeling budget
--selection_radius SELECTION_RADIUS Selection radius
--reward_threshold REWARD_THRESHOLD Reward threshold
--w2v                               Use w2v embeddings
```

### MNIST
```
--hidden_size HIDDEN_SIZE           Size of hidden layer in deep RL
--episodes EPISODES                 Number of episodes
--learning_rate_rl LEARNING_RATE_RL Learning rate
--margin MARGIN                     Rank loss margin.
--num_epochs NUM_EPOCHS             Number of training epochs.
--full_epochs FULL_EPOCHS           Number of training epochs.
--init_samples INIT_SAMPLES         number of random inital training data
--batch_size BATCH_SIZE             Size of a training mini-batch.
--budget BUDGET                     Our labeling budget
--selection_radius SELECTION_RADIUS Selection radius
--reward_threshold REWARD_THRESHOLD Reward threshold
--w2v                               Use w2v embeddings
```

### Digits
```
--hidden_size HIDDEN_SIZE           Size of hidden layer in deep RL
--episodes EPISODES                 Number of episodes
--learning_rate_rl LEARNING_RATE_RL learning rate
--margin MARGIN                     Rank loss margin.
--num_epochs NUM_EPOCHS             Number of training epochs.
--full_epochs FULL_EPOCHS           Number of training epochs.
--init_samples INIT_SAMPLES         Number of random inital training data
--batch_size BATCH_SIZE             Size of a training mini-batch.
--budget BUDGET                     Our labeling budget
--selection_radius SELECTION_RADIUS Selection radius
--reward_threshold REWARD_THRESHOLD Reward threshold
--w2v                               Use w2v embeddings
```

## Implementation of custom datasets
To implement and train the agent on your own datasets, create a folder within `datasets` with the following files:

- `dataset.py `
Contains a single function, `load_data()`, that is responsible for loading the dataset and returning one tuple for train_data, dev_data and test_data, in that order. Each tuple is on the form (x, y), where the label for x[0] is y[0], and so on. Import opt from config, set the following properties of it
    - `data_sizes` Length of each element in the state vector presented to the reinforcement agent
    - `data_len` Length of the training data, e.g. `len(train_data[0])`


- `model.py`
Has to implement the following functions
    - `reset(self)` Reset the models parameters to initial values
    - `train_model(self, train_data, epochs)` Train the model using labeled data. Typically what you previously added in `data['active']`.
    - `validate(self, data)` Fast validation function that validates `data`, and will be run to determine reward used for the reinforcement agent.
        - Returns a dictionary with the following required items
            - `performance` What to use to measure increase in performance. Higher performance is beneficial, so if using something negative, e.g. loss, as performance, negate it
    - `performance_validate(self, data)` Can be a more heavier performance validation function that only runs at the end of each episode.
    - `get_state(self, index)` Given index, calculate the state for the reinforcement agent using `data['train'][0][index]`.
    - `query(self, index)` 'Label' the current datapoint in the stream. Typically appends `data['train_deleted'][0][index]` to `data['active'][0]` and `data['train_deleted'][1][index]` to `data['active'][1]`. If wanted, add other indices, maybe by computing similarity measures. Has to return which indices were added, so they can be removed from the stream.
    - `encode_episode_data(self)` If necessary, perform computation here so that you don't have to do it every time in `get_state()`. This method is called each time the model is trained, so that whatever you calculate is representative of the latest model state.

- `__init__.py`
Include logic for exposing the previous 2 files. Has to expose the `load_data()` and the model described above.
