import numpy as np
import torch
import my_dataset
import torch.utils.data.sampler as sampler
import random


class MySampler(sampler.Sampler):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """


    def __init__(self, data_set,classes_per_it, num_samples, iterations,max_samples_per_class):
        # super(MySampler, self).__init__()
        self.count = 0
        self.dataset = data_set
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.dataset.y, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.dataset.y))
        self.indexes = np.empty((len(self.classes), max_samples_per_class), dtype=int) * np.nan
        # self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        self.label_count = 100000
        for idx, label in enumerate(self.dataset.y):
            # if(label ==95):
            #     print(123)
            label_idx = np.argwhere(self.classes == label).item()
            if (self.label_count != label):
                self.count = 0
            if(self.count<max_samples_per_class):
                self.label_count = label
                self.indexes[label_idx, self.count] = idx
                self.count +=1
                self.numel_per_class[label_idx] += 1


            # self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            # self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
                yield a batch of indexes
                '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        # 每个iteration 拿出 cpi个class 2个status 每个spc个sample
        # self.dataset.y unique 取前一半class random 5类下标
        # 从self.dataset.classes 的tuple中搜寻对应腐烂的5类的下标 总10个下标
        # 从self.indexes中根据下标所在行每类各随机抽取10个
        # 2个(5,10,3,100,100)stack为一个batch
        #
        for it in range(self.iterations):
            batch_size = spc * cpi * 2
            batch = np.array([])
            unique_class = np.unique(self.dataset.y)
            ripe_class = unique_class[:int(len(unique_class)/2)]
            ripe_class_cpi = random.sample(list(ripe_class),cpi)
            for ripe_line_id in ripe_class_cpi:
                food_name = self.dataset.all_classes[ripe_line_id][0]
                tup = (food_name, 'rotten')
                rotten_line_id = self.dataset.all_classes.index(tup)
                # rotten_line_id = 95
                rotten_food_list = self.indexes[rotten_line_id].numpy()
                rotten_food_cpi = np.random.choice(rotten_food_list, spc, replace=False)
                ripe_food_list = self.indexes[ripe_line_id].numpy()
                ripe_food_cpi = np.random.choice(ripe_food_list,spc,replace=False)
                batch = np.concatenate([batch,ripe_food_cpi,rotten_food_cpi])
            # batch = torch.from_numpy(batch[torch.randperm(len(batch))]).int()
            batch = torch.from_numpy(batch).int()
            yield batch


    def __len__(self):
        return self.iterations



def main():
    np.random.choice(5, 3, replace=False)
    dataset = my_dataset.MyDataset()
    sampler = MySampler(dataset,5,5,5)
    for item in sampler:
        print(item)
    # a = sampler.__iter__()

if __name__ == '__main__':
    main()