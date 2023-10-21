import random
import math
import os
from itertools import cycle

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import time

import torch.distributed as dist




class TestDataset(IterableDataset):
    def __init__(self, batch_size=16, drop_last=True):

        self.wsi = []
        # for i in range(0, 300 * 100, 100):
        for i in range(0, 100 * 20, 20):
            # self.wsi.append(range(i, i + random.randint(80, 100)))
            # self.wsi.append(range(i, i + random.randint(20, 30 - 1)))
            self.wsi.append(range(i, i + 20))

        # self.begin = 0
        self.batch_size = batch_size # batch_size for each dataset
        self.drop_last = drop_last

        self.start_idx = 0
        # self.end_idx = self.batch_size - 1
        self.end_idx = len(self.wsi) - 1

        # self.start_idx - self.end_idx = the sub wsis for this dataset to process

    def __iter__(self):
        # self.wsi
        # random.shuffle(self.wsi)
        # worker_info = torch.utils.data.get_worker_info()
        # print(worker, 'workers............................................', worker.num_workers)

        # single process
        # if worker_info is not None:
            # per_worker = int(math.ceil((self.end_idx - self.start_idx) / float(worker_info.num_workers)))
            # worker_id = worker_info.id
            # iter_start = self.start_idx + worker_id * per_worker
            # iter_end = min(iter_start + per_worker, self.end)
            # self.start_idx = iter_start
            # self.end_idx = iter_end

        # return iter(range(iter_start, iter_end))





        # for idx in range(0, len(self.wsi), self.batch_size):
        # for idx in range(self.start_idx, self.end_idx + 1, self.batch_size):
        for idx in range(self.start_idx, self.end_idx, self.batch_size):
            if not idx + self.batch_size <= self.end_idx:
                return

            batch_wsi = self.wsi[idx : idx + self.batch_size]
            print('batch_wsi:', len(batch_wsi), idx, idx + self.batch_size)
            # batch_wsi = self.wsi[idx + self.start_idx: idx + self.end_idx - self.start_idx]
            # print('idx.................', idx)


            # print('len________', len(batch_wsi))

            if self.drop_last:
                if len(batch_wsi) != self.batch_size:
                # if len(batch_wsi) != self.start_idx - self.end_idx:
                    return
            else:
                for i in range(len(batch_wsi), self.batch_size):
                # for i in range(len(batch_wsi), self.start_idx - self.end_idx):
                    batch_wsi.append(self.wsi[self.start_idx + i])

            assert len(batch_wsi) == self.batch_size
            # assert len(batch_wsi) == self.start_idx - self.end_idx
            max_batch_lenth = max([len(x) for x in batch_wsi])

            batch_wsi = [cycle(x) for x in batch_wsi]
            for i in range(max_batch_lenth):
                # print('ccccccccccc', i)
                sleep_time = 0.005 * len(batch_wsi)
                time.sleep(sleep_time)

                yield [next(x) for x in batch_wsi]

            print('hello????????????????????????????????')
            yield [-1] * self.batch_size
            # yield [-1] * (self.start_idx - self.end_idx)

        # self.begin += self.batch_size





        # for i
class TestDistIdeas:
    def __init__(self):



        if 'LOCAL_WORLD_SIZE' not in os.environ:
                # raise RuntimeError("Requires distributed package to be available")

            ds1 = TestDataset(batch_size=16)
            # print(ds1.start_idx, ds1.end_idx, ds1.batch_size)
            ds1.start_idx = 0
            # ds1.end_idx = 0 + int(len(ds1.wsi) / 2)  # works
            # ds1.end_idx = 0 + int(len(ds1.wsi) / 2)
            # ds1.batch_size = 8
            print(ds1.start_idx, ds1.end_idx, ds1.batch_size)
            # import sys; sys.exit()


            #ds2 = TestDataset(batch_size=16)
            #ds2.start_idx = 0
            #ds2.end_idx = 0 + int(len(ds2.wsi) / 2)
            #ds2.batch_size = 8

            self.datasets = [
                # ds1, ds2
                ds1
            ]

        else:
            num_replicas = int(os.environ['LOCAL_WORLD_SIZE'])
            rank = int(os.environ['LOCAL_RANK'])
            # print(type(rank), type(num_replicas))
            num_samples = int(100 / num_replicas)

            ds1 = TestDataset(batch_size=16)
            # print(ds1.start_idx, ds1.end_idx, ds1.batch_size)
            ds1.start_idx = rank * num_samples
            ds1.end_idx = (rank + 1)* num_samples
            # ds1.end_idx = 0 + int(len(ds1.wsi) / 2)  # works
            # ds1.end_idx = 0 + int(len(ds1.wsi) / 2)
            # ds1.batch_size = 8
            print(ds1.start_idx, ds1.end_idx, ds1.batch_size)
            # import sys; sys.exit()


            #ds2 = TestDataset(batch_size=16)
            #ds2.start_idx = 0
            #ds2.end_idx = 0 + int(len(ds2.wsi) / 2)
            #ds2.batch_size = 8

            self.datasets = [
                # ds1, ds2
                ds1
            ]



    def __iter__(self):
        for batch_parts in zip(*[DataLoader(dataset, num_workers=1, batch_size=None) for dataset in self.datasets]):
            # print(batch_parts)
            yield batch_parts



# print(os.environ['MASTER_ADDR'])
# print(os.environ['LOCAL_WORLD_SIZE'])
# import sys; sys.exit()

import time
s1 = time.time()
dataloader = TestDistIdeas()
# for ds in TestDistIdeas():
for ds in dataloader:
    # print(ds, os.environ['LOCAL_RANK'])
    print(ds)

# ds = TestDataset(drop_last=False)
# dl = DataLoader(ds, batch_size=None, num_workers=4)

# for i in dl:
#     print(i)

# for ds in TestDistIdeas():
#      print(ds)
s2 = time.time()
print(s2 - s1)