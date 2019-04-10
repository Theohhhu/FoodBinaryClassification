import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import re
from PIL import Image
import numpy as np


IMG_CACHE = {}

class MyDataset(data.Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, mode):
        if mode=='train':
            self.root= 'C:\\Users\\HASEE\\PycharmProjects\\datasets\\train'
            # self.root= 'C:\\Users\\HASEE\\PycharmProjects\\fruit_datasets\\train'

        if mode == 'dev':
            self.root = 'C:\\Users\\HASEE\\PycharmProjects\\datasets\\dev'
            # self.root= 'C:\\Users\\HASEE\\PycharmProjects\\fruit_datasets\\dev'

        self.ripe_classes,self.rotten_classes,self.raw_classes = get_current_classes(self.root)
        self.items_raw,self.items_rotten,self.items_ripe = find_items(self.root)

        # self.idx_classes_raw = index_classes(self.items_raw)
        # self.idx_classes_rotten = index_classes(self.items_rotten)
        # self.idx_classes_ripe = index_classes(self.items_ripe)

        self.all_items = self.items_ripe+self.items_rotten
        self.all_classes = self.ripe_classes + self.rotten_classes
        self.idx_classes = index_classes(self.all_items)

        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self.all_items))])

        # paths_raw, self.raw_y = zip(*[self.get_path_label(pl,'raw')
        #                       for pl in range(len(self.items_raw))])
        # paths_rotten, self.rotten_y = zip(*[self.get_path_label(pl,'rotten')
        #                       for pl in range(len(self.items_rotten))])
        # paths_ripe, self.ripe_y = zip(*[self.get_path_label(pl,'ripe')
        #                       for pl in range(len(self.items_ripe))])
        #
        # self.raw_x = map(load_img, paths_raw, range(len(paths_raw)))
        # self.rotten_x = map(load_img, paths_rotten, range(len(paths_rotten)))
        # self.ripe_x = map(load_img, paths_ripe, range(len(paths_ripe)))

        self.x = map(load_img, paths, range(len(paths)))
        self.x = list(self.x)

        print("initialized")

        # self.raw_x = list(self.raw_x)
        # self.rotten_x = list(self.rotten_x)
        # self.ripe_x = list(self.ripe_x)



    def __getitem__(self, index):
        x = self.x[index]
        # if self.transform:
        #     x = self.transform(x)
        return x, self.y[index]

    def __len__(self):
        raise NotImplementedError

    def get_path_label(self, index):
        # items=[]
        # if(status == 'raw'):
        #     items = self.items_raw
        #     idx_classes = self.idx_classes_raw
        # if (status == 'rotten'):
        #     items = self.items_rotten
        #     idx_classes = self.idx_classes_rotten
        # if (status == 'ripe'):
        #     items = self.items_ripe
        #     idx_classes = self.idx_classes_ripe


        filename = self.all_items[index][0]
        img = str.join('\\', [self.all_items[index][2], filename])
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]

        return img, target

def find_items(root_dir):

    retour_ripe = []
    retour_raw = []
    retour_rotten = []

    for root, dirs, files in os.walk(root_dir):

        if (root.endswith('ripe') or root.endswith('rotten') or root.endswith('raw')):
            print(root)
            print(files)
            for f in files:
                label = re.search(r'datasets\\.*\\(.*)\\.*$', root).group(1)
                status = re.search(r'datasets\\.*\\.*\\(.*)$', root).group(1)
                if(status == 'ripe'):
                    retour_ripe.extend([(f, label, root, status)])
                if (status == 'rotten'):
                    retour_rotten.extend([(f, label, root, status)])
                if (status == 'raw'):
                    retour_raw.extend([(f, label, root, status)])

    return retour_raw,retour_rotten,retour_ripe

def get_current_classes(root_dir):
    num_class_ripe = []
    num_class_rotten = []
    num_class_raw = []

    for root, dirs, files in os.walk(root_dir):
        # begin
        # if root ends with ripe or rotten
        # regex get datasets/(.*)$ as class
        if (root.endswith('ripe') or root.endswith('rotten') or root.endswith('raw')):
            label = re.search(r'datasets\\.*\\(.*)\\.*$', root).group(1)
            status = re.search(r'datasets\\.*\\.*\\(.*)$', root).group(1)
            if (status == 'ripe'):
                num_class_ripe.extend([(label, status)])
            if (status == 'rotten'):
                num_class_rotten.extend([(label, status)])
            if (status == 'raw'):
                num_class_raw.extend([(label, status)])
    return num_class_ripe,num_class_rotten,num_class_raw

def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] + i[-1] in idx):
            idx[i[1] + i[-1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx

def load_img(path, idx):
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x
    if x.mode != 'RGB':
        x = x.convert('RGB')
    x = x.resize((100, 100))
    # if(idx==360):
    #     print("xx")
    shape = 3, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x)
    x = x.transpose(0, 1).contiguous().view(shape)

    return x

def main():

    dataset = MyDataset()

if __name__ == '__main__':
    main()


