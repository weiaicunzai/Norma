from itertools import cycle, chain

import torch

from torch.utils.data import IterableDataset, DataLoader, Dataset


import random


class MyTest(IterableDataset):

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    @property
    def shuffle_data_list(self):
        # print(self.data)
        # print(random.sample(self.data, len(self.data)))
        return random.sample(self.data, len(self.data))
        # return self.data
        # return [d for d in self.data]

    def process_data(self, data):
        for x in data:
            yield x

    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffle_data_list) for _ in range(self.batch_size)])

    def __iter__(self):
        # return cycle(self.get_stream(self.data))
        return self.get_streams()

import time

class  MapDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        worker = torch.utils.data.get_worker_info()
        # print(11, worker)
        worker_id = worker.id if worker is not None else -1

        start = time.time()
        time.sleep(0.1)
        end = time.time()

        return self.data[idx], worker_id, start, end

data = [
    range(5, 10),
    range(50, 100),
    range(500, 1000),
]

# a = cycle(data)
# print(a)
# for i in a :
#     print(i)
# import sys; sys.exit()
iterable_dataset = MyTest(data, 4)

# loader = DataLoader(iterable_dataset, batch_size=4)
loader = DataLoader(iterable_dataset, batch_size=None)

map_dataset = MapDataset(data=range(20))
loader = DataLoader(map_dataset, batch_size=4, num_workers=0)

count = 0
for i in loader:
    print(i)

    count += 1

    if count > 10:
        break