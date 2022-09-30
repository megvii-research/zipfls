import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import torchvision.transforms as transforms
import numpy as np
import settings

CARD_NUM = 4

if torch.cuda.is_available():
    CARD_NUM = torch.cuda.device_count()


class FriendSampler(Sampler):
    def __init__(self, dataset, friend_num=1):
        assert friend_num >= 1
        self.dataset = dataset
        self.friend_num = friend_num
        self.pair_num = self.friend_num + 1
        #assert self.pair_num % 2 == 0
        assert settings.minibatch_size % self.pair_num == 0
        assert settings.minibatch_size // CARD_NUM % self.pair_num == 0, 'more than friend expected in gpu might occur'
        self.sample_num = len(self.dataset.samples)
        self.sample_class_num_each = settings.minibatch_size // self.pair_num
        assert self.sample_class_num_each <= 1000

    def __len__(self):
        return self.sample_num

    def __iter__(self):
        list_of_clsidxs = []

        for label in self.dataset.cls_idxs:
            idxs = list(self.dataset.cls_idxs[label])
            np.random.shuffle(idxs)
            list_of_clsidxs.append(idxs)

        indexes = []
        total_len = len(self.dataset)
        remains = []
        while len(list_of_clsidxs) > 0:
            cls_indexs = np.random.permutation(len(list_of_clsidxs))[
                :self.sample_class_num_each]
            remove_cls = []
            for cls_index in cls_indexs:
                cur = list_of_clsidxs[cls_index]
                if len(cur) < self.pair_num:
                    remains.extend(cur)
                    cur = []
                else:
                    indexes.extend(cur[-self.pair_num:])
                    # pop from last would be faster
                    for i in range(self.pair_num):
                        cur.pop()
                if len(cur) == 0:
                    remove_cls.append(cls_index)

            # delete empty class list, delete element from tail would be faster
            if len(remove_cls) > 0:
                remove_cls.sort()
                for i in remove_cls[::-1]:
                    del list_of_clsidxs[i]

        indexes.extend(remains)
        assert len(indexes) == total_len, (len(indexes), total_len)

        return iter(indexes)


class PairSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.samples = self.dataset.samples
        self.idx_clses = self.dataset.idx_clses

    def __iter__(self):
        new_indices = []
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.idx_clses[idx]
                pair_indices.append(random.choice(
                    self.dataset.cls_idxs[y]))
            new_indices.extend(batch_indices + pair_indices)
        print('real dataset size is ', len(new_indices))
        return iter(new_indices)

    def __len__(self):
        return len(self.samples) * 2


if __name__ == '__main__':
    from dataset import ImageNetDataset
    import time
    from tqdm import tqdm

    def check_cls_balance(idxs, ds):
        step = settings.minibatch_size // CARD_NUM
        i = 0
        with tqdm(total=len(idxs)) as pbar:
            while i < len(idxs):
                print(i)
                cur = list(map(lambda x: ds[x][1], idxs[i:i+step]))
                l1 = len(cur) // 2
                l2 = len(set(cur))
                assert l1 == l2, (l1, l2)
                i += step
                pbar.update(step)

    ds = ImageNetDataset(True)
    print(len(ds))

    friend_num = 1
    it_1 = FriendSampler(ds, friend_num)
    it_2 = FriendSampler_2(ds, friend_num)
    for i in range(160):
        t0 = time.time()
        idxs = list(iter(it_1))
        check_cls_balance(idxs, ds)
        print(i, len(idxs), len(set(idxs)), idxs[:10], list(
            map(lambda x: ds[x][1], idxs[:10])), 'it1 time', time.time() - t0)

        t0 = time.time()
        idxs = list(iter(it_2))
        print(i, len(idxs), len(set(idxs)), idxs[:10], list(
            map(lambda x: ds[x][1], idxs[:10])), 'it2 time', time.time() - t0)

        print('-' * 100)
    print('done')
