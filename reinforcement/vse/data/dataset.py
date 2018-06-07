import torch
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
from sklearn import datasets, svm, metrics
# from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
import sklearn

from config import data

def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids

class FlickrDataset(torch.utils.data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)


class PrecompDataset(torch.utils.data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, vocab, data_length):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            # self.length = 5000
            # self.length = 1000
            self.length = data_length

    def delete_indices(self, indices):
        self.images = np.delete(self.images, indices, axis=0)
        self.captions = np.delete(self.captions, indices, axis=0)
        self.length = len(self.captions)

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length

    def shuffle(self):
        pass


class MRDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_split):
        x, y = [], []
        with open("{}/MR/rt-polarity.pos".format(data_path), "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                x.append(line.split())
                y.append(1)

        with open("{}/MR/rt-polarity.neg".format(data_path), "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                x.append(line.split())
                y.append(0)

        x, y = sklearn.utils.shuffle(x, y)
        dev_idx = len(x) // 10 * 8
        test_idx = len(x) // 10 * 9

        words = sorted(list(set([w for sent in x for w in sent])))
        # print(len(words))
        self.vocab = {w: i for i, w in enumerate(words)}
        data.vocab = {w: i for i, w in enumerate(words)}
        
        if data_split == 'train':
            self.sentences = x[:dev_idx]
            self.targets = y[:dev_idx]
        elif data_split == 'dev':
            self.sentences = x[dev_idx:test_idx]
            self.targets = y[dev_idx:test_idx]
        elif data_split == 'test':
            self.sentences = x[test_idx:]
            self.targets = y[test_idx:]

        self.length = len(self.sentences)


    def shuffle(self):
        self.sentences, self.targets = sklearn.utils.shuffle(self.sentences, self.targets)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        target = self.targets[index]
        tokens = [self.vocab[word] for word in sentence]
        padding = (59 - len(sentence)) * [len(self.vocab)]
        tokens_padded = tokens + padding
        return tokens_padded, target


    def __len__(self):
        return self.length

    def shuffle(self):
        pass

class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_split):
                
        # The digits dataset
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))
        target = digits.target
        
        data, target = sklearn.utils.shuffle(data, target)
        
        dev_idx = n_samples // 4 #TODO correct? 
        test_idx = n_samples // 2

        if data_split == 'train':
            self.images = data[:dev_idx]
            self.targets = target[:dev_idx]
        elif data_split == 'dev':
            self.images = data[dev_idx:test_idx]
            self.targets = target[dev_idx:test_idx]
        elif data_split == 'test':
            self.images = data[test_idx:]
            self.targets = target[test_idx:]

        self.length = len(self.images)


    def shuffle(self):
        self.images, self.targets = sklearn.utils.shuffle(self.images, self.targets)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]
        # tokens = [self.vocab[word] for word in image] #OK? 
        # padding = (59 - len(image)) * [len(self.vocab)]
        # tokens_padded = tokens + padding
        return image, target

    def __len__(self):
        return self.length

    def shuffle(self):
        pass


class ActiveDataset(torch.utils.data.Dataset):
    """
    Initially empty dataset to contain the train
    data used for active learning.
    """

    def __init__(self):
        self.captions = []
        self.images = []
        self.length = len(self.captions)

    def __getitem__(self, index):
        image = self.images[index]
        caption = self.captions[index]
        return image, caption

    def __len__(self):
        # return self.length
        return len(self.captions)

    def add_single(self, image, caption):
        self.images.append(image)
        self.captions.append(caption)
        self.length = len(self.images)

    def add_multiple(self, images, captions):
        self.images.extend(images)
        self.captions.extend(captions)
        # self.length = len(self.captions)

    def shuffle(self):
        self.images, self.captions = sklearn.utils.shuffle(self.images, self.captions)

def collate_fn_mr(data):
    sentences, targets = zip(*data)
    sentences, targets = list(sentences), list(targets)
    sentences = torch.stack(torch.LongTensor(sentences))
    targets = torch.LongTensor(targets)
    return sentences, targets

#TODO why is this here? 
def collate_fn_digit(data):
    images, targets = zip(*data)
    images, targets = list(images), list(targets)
    # images = torch.stack(torch.LongTensor(images))
    # targets = torch.LongTensor(targets)
    return images, targets


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=collate_fn):
    # """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # if 'coco' in data_name:
    #     # COCO custom dataset
    #     dataset = CocoDataset(root=root,
    #                           json=json,
    #                           vocab=vocab,
    #                           transform=transform, ids=ids)
    if 'f8k' in data_name or 'f30k' in data_name:
        dataset = FlickrDataset(root=root,
                                split=split,
                                json=json,
                                vocab=vocab,
                                transform=transform)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=torch.cuda.is_available(),
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_mr_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2, data_length=100):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = MRDataset(data_path, data_split)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=torch.cuda.is_available(),
                                              collate_fn=collate_fn_mr)
    return data_loader

def get_digit_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2, data_length=100):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = DigitDataset(data_path, data_split)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=torch.cuda.is_available(),
                                              collate_fn=collate_fn_digit)
    return data_loader

def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2, data_length=100):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab, data_length)


    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=torch.cuda.is_available(),
                                              collate_fn=collate_fn)
    return data_loader

def get_active_loader(batch_size=100, shuffle=True, num_workers=2):
    dset = ActiveDataset()

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=torch.cuda.is_available(),
                                              collate_fn=collate_fn_mr)
    return data_loader

# def get_episode_loader(vocab, batch_size=100, shuffle=True, num_workers=2):
#     dset = ActiveDataset(vocab)
#
#     data_loader = torch.utils.data.DataLoader(dataset=dset,
#                                               batch_size=batch_size,
#                                               shuffle=shuffle,
#                                               pin_memory=True,
#                                               collate_fn=collate_fn)
#     return data_loader



def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    active_loader = get_active_loader()
    if opt.dataset == 'vse':
        dpath = os.path.join(opt.data_path, data_name)
        if opt.data_name.endswith('_precomp'):
            train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                              batch_size, True, workers)
            val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                            batch_size, False, workers, data_length=opt.val_size)
            val_tot_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                            batch_size, False, workers, data_length=5000)
        else:
            # Build Dataset Loader
            roots, ids = get_paths(dpath, data_name, opt.use_restval)

            transform = get_transform(data_name, 'train', opt)
            train_loader = get_loader_single(opt.data_name, 'train',
                                             roots['train']['img'],
                                             roots['train']['cap'],
                                             vocab, transform, ids=ids['train'],
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=workers,
                                             collate_fn=collate_fn)

            transform = get_transform(data_name, 'val', opt)
            val_loader = get_loader_single(opt.data_name, 'val',
                                           roots['val']['img'],
                                           roots['val']['cap'],
                                           vocab, transform, ids=ids['val'],
                                           batch_size=batch_size, shuffle=False,
                                           num_workers=workers,
                                           collate_fn=collate_fn)

    elif opt.dataset == "MR" :
        train_loader = get_mr_loader(opt.data_path, 'train', vocab, opt,
                                          batch_size, True, workers)
        val_loader = get_mr_loader(opt.data_path, 'dev', vocab, opt,
                                        batch_size, False, workers)
        val_tot_loader = get_mr_loader(opt.data_path, 'dev', vocab, opt,
                                        batch_size, False, workers)

    elif opt.dataset == "digit":
        train_loader = get_digit_loader(opt.data_path, 'train', vocab, opt,
                                          batch_size, True, workers)
        val_loader = get_digit_loader(opt.data_path, 'dev', vocab, opt,
                                        batch_size, False, workers)
        val_tot_loader = get_digit_loader(opt.data_path, 'dev', vocab, opt,
                                        batch_size, False, workers)


    return active_loader, train_loader, val_loader, val_tot_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                         batch_size, False, workers)
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, split_name, opt)
        test_loader = get_loader_single(opt.data_name, split_name,
                                        roots[split_name]['img'],
                                        roots[split_name]['cap'],
                                        vocab, transform, ids=ids[split_name],
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=workers,
                                        collate_fn=collate_fn)

    return test_loader
