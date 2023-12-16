import random
from typing import Iterator
import warnings
# from itertools import cycle
# from utils.mics import cycle
from torch.utils.data import IterableDataset
import torch
import cv2
# default_collate

from PIL import Image


# class IterableMixIn:
#     def return_iter(wsis, batch_size):
#         max_len

class WSIDataset(IterableDataset):
    def __init__(self, wsis, batch_size, transforms=None):
        """the num_worker of each CAMLON16 dataset is one, """

        self.wsis = wsis

        self.batch_size = batch_size
        assert len(self.wsis) % self.batch_size == 0

        self.global_seq_len = self.cal_seq_len()
        self.trans = transforms


    def shuffle(self):
        """manually shuffle all the wsis, because when num_workers > 0,
        the copy of dataset wont update in the main process """
                # if self.shuffle:
        random.shuffle(self.wsis)
            # self.global_seq_len = self.get_global_seq_len()

    # @property
    def cal_seq_len(self):
    # def global_seq_len(self):
        outputs = []
        for wsi in self.wsis:
            outputs.append(wsi.num_patches)

        # print('cccccc', outputs, id(self))
        return outputs

    def cycle(self, iterable):
        while True:
            for data in iterable:
                yield data

    def __iter__(self):

        for idx in range(0, len(self.wsis), self.batch_size):

            batch_wsi = self.wsis[idx : idx + self.batch_size]


            # if self.drop_last:
            #     if len(batch_wsi) != self.batch_size:
            #         return
            # else:
            #     for i in range(len(batch_wsi), self.batch_size):
            #     # for i in range(len(batch_wsi), self.start_idx - self.end_idx):
            #         batch_wsi.append(self.wsis[i])


            assert len(batch_wsi) == self.batch_size

            # max_batch_lenth = max([len(x) for x in batch_wsi])

            # print([x.wsi_name for x in batch_wsi])
            batch_wsi = [self.cycle(x) for x in batch_wsi]
            # for i in range(self.max_batch_len):

            max_len_idx = idx // self.batch_size
            if not self.global_seq_len[max_len_idx]:
                warnings.warn('max batch len equals 0')
                continue
            # for _ in  range(self.global_seq_len[idx // self.batch_size]):
            max_len = self.global_seq_len[max_len_idx]
            for patch_id in  range(max_len):
                # print('ccccccccccc', i)
                # sleep_time = 0.005 * len(batch_wsi)
                # time.sleep(sleep_time)

                # try:

                    # print(batch_wsi)
                    # print(1111, self.global_seq_len[idx // self.batch_size], idx // self.batch_size, _)
                    # outputs = [next(x) for x in batch_wsi]
                    outputs = []
                    for x in batch_wsi:
                        data = next(x).copy()
                        # data = {
                            # 'img': cv2.imread('test_512_patch.jpg'),
                            # 'img': Image.open('test_512_patch.jpg'),
                            # 'label': 1
                        # }
                        # print(type(data['img']))
                        # if self.trans is not None:
                        #     data['img'] = self.trans(data['img'])

                        # data = {
                        #     'img': torch.randn((3, 256, 256)),
                        #     'label': torch.randn((3, 256, 256)),
                        # }
                        if patch_id < max_len - 1:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        # print(patch_id, max_len - 1, data['is_last'])

                        outputs.append(data)

                        if self.trans is not None:
                            # print('heheheh')
                            data['img'] = self.trans(image=data['img'])['image'] # A

                    yield outputs


import time
import random
class Test(IterableDataset):
    def __init__(self):
        super().__init__()
        self.wsis = [
            [
                iter(range(0, 100)),
                iter(range(10, 200)),
            ],
            [
                iter(range(20, 300)),
                iter(range(30, 400))
            ],
        ]

        # return

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        # total_workers = worker_info.num_workers
        # print(worker_id)


        for wsis in self.wsis:
            for taa in zip(*wsis):
                # for i in taa:
                if worker_id == 0:
                    time.sleep(random.random() / 10)

                if worker_id == 1:
                    time.sleep(random.random() / 10)
                    # time.sleep(random.random() / 10)



                yield taa[worker_id], worker_id

import math
class MyIterableDataset(torch.utils.data.IterableDataset):
     def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

        self.wsis = [
            # [
                # iter(range(0, 100)),
                # iter(range(10, 110)),
                # iter(range(20, 120)),
                # iter(range(30, 130)),
                # iter(range(40, 140)),
                # iter(range(50, 150)),
                # iter(range(60, 160)),
                # iter(range(70, 170)),
                # iter(range(80, 180)),
                list(range(0, 100)),
                list(range(10, 110)),
                list(range(20, 120)),
                list(range(30, 130)),
                list(range(40, 140)),
                list(range(50, 150)),
                list(range(60, 160)),
                list(range(70, 170)),
        ]

     def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = self.start
             iter_end = self.end
        else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)

        #  iter(range(iter_start, iter_end))
        #  range(self.start + worker_id * , self.end, )
        # #  for i in iter(range(iter_start, iter_end)):
        #  for i in iter(range(iter_start, iter_end)):
        #     time.sleep(random.random() / 10)
        #     yield i, worker_id
        # step = 8
        # iter =
        count = 0
        current_tmp = []
        bs = 8
        for i in range(self.start, self.end):
            batch_idx = int(i / bs)
            if batch_idx %  worker_info.num_workers == worker_id:
                current_tmp.append(i)


            # count += 1
            # if count == 8:
            # count %
            # worker_id * 8

        count = 0
        tmp = []
        # print(current_tmp)
        for i in current_tmp:
            count += 1
            # time.sleep(random.random() / 10)
            time.sleep(0.003)
            tmp.append(i)
            # print(count)
            if count == bs:
                count = 0
                yield tmp
                tmp = []

        # for start_idx in range(0, len(self.wsis), step=8):
            # for wsis in self.wsis[start_idx: start_idx+step]:

class MyMapDataset(torch.utils.data.Dataset):
     def __init__(self, start, end):
        super().__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

        self.wsis = [
            # [
                # iter(range(0, 100)),
                # iter(range(10, 110)),
                # iter(range(20, 120)),
                # iter(range(30, 130)),
                # iter(range(40, 140)),
                # iter(range(50, 150)),
                # iter(range(60, 160)),
                # iter(range(70, 170)),
                # iter(range(80, 180)),
                list(range(0, 100)),
                list(range(10, 110)),
                list(range(20, 120)),
                list(range(30, 130)),
                list(range(40, 140)),
                list(range(50, 150)),
                list(range(60, 160)),
                list(range(70, 170)),
        ]
        tmp = []
        for wsi in self.wsis:
            for w in wsi:
                tmp.append(w)
        self.wsis = tmp

     def __len__(self):
         return len(self.wsis)

     def __getitem__(self, idx):
        num = self.wsis[idx]

        # time.sleep(random.random() / 10)
        time.sleep(0.003)
        return num



     #def __iter__(self):
     #   worker_info = torch.utils.data.get_worker_info()
     #   if worker_info is None:  # single-process data loading, return the full iterator
     #        iter_start = self.start
     #        iter_end = self.end
     #   else:  # in a worker process
     #        # split workload
     #        per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
     #        worker_id = worker_info.id
     #        iter_start = self.start + worker_id * per_worker
     #        iter_end = min(iter_start + per_worker, self.end)

     #   #  iter(range(iter_start, iter_end))
     #   #  range(self.start + worker_id * , self.end, )
     #   # #  for i in iter(range(iter_start, iter_end)):
     #   #  for i in iter(range(iter_start, iter_end)):
     #   #     time.sleep(random.random() / 10)
     #   #     yield i, worker_id
     #   # step = 8
     #   # iter =
     #   count = 0
     #   current_tmp = []
     #   bs = 8
     #   for i in range(self.start, self.end):
     #       batch_idx = int(i / bs)
     #       if batch_idx %  worker_info.num_workers == worker_id:
     #           current_tmp.append(i)


     #       # count += 1
     #       # if count == 8:
     #       # count %
     #       # worker_id * 8

     #   count = 0
     #   tmp = []
     #   # print(current_tmp)
     #   for i in current_tmp:
     #       count += 1
     #       time.sleep(random.random() / 10)
     #       tmp.append(i)
     #       # print(count)
     #       if count == bs:
     #           count = 0
     #           yield tmp
     #           tmp = []

# a = MyIterableDataset(start=0, end=400)
# b = MyMapDataset(start=0, end=400)

# def to_nothing(input):
    # return input
# dataloader = torch.utils.data.DataLoader(a, num_workers=4, batch_size=None)
# dataloader = torch.utils.data.DataLoader(b, num_workers=4, batch_size=8, collate_fn=to_nothing)
# for i in a:

class MMM(torch.utils.data.Dataset):
    def __init__(self, start, end):
        super().__init__()
        self.data = range(start, end)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        time.sleep(0.003)
        return data

class Iterable(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super().__init__()
        self.data = list(range(start, end))
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = self.start
             iter_end = self.end
        else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)

        # print(iter_start, iter_end)
        data = self.data[iter_start : iter_end]

        for i in data:
            time.sleep(0.003)
            yield i




map_dataset = MMM(0, 10000)
iter_dataset = Iterable(0, 10000)
dataloader = torch.utils.data.DataLoader(map_dataset, num_workers=4, batch_size=64)
# dataloader = torch.utils.data.DataLoader(iter_dataset, num_workers=4, batch_size=64)


import time
t1 = time.time()
for i in dataloader:
    # print(i)
    pass

print(time.time() - t1)