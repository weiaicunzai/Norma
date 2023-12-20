import random
import warnings
# from itertools import cycle
# from utils.mics import cycle
from torch.utils.data import IterableDataset, default_collate
import torch
import cv2
from torch.utils.data import default_collate
# default_collate

from PIL import Image
import lmdb
import numpy as np

import time
import math

from .wsi_reader import CAMLON16MixIn

class WSIDatasetNaive(IterableDataset):
    def __init__(self, data_set, lmdb_path, batch_size, drop_last=False, allow_reapt=False, transforms=None, dist=None):
        """the num_worker of each CAMLON16 dataset is one, """
        assert data_set in ['train', 'val']


        self.batch_size = batch_size
        self.drop_last = drop_last
        self.allow_reapt = allow_reapt
        self.data_set = data_set
        self.trans = transforms

        self.orig_wsis = self.get_wsis(data_set=data_set)
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.seed = 52

        self.dist = dist

    def split_wsis(self, wsis):
        if not self.dist.is_initialized():
            return wsis
        else:
            """get wsis for each gpu"""
            rank = self.dist.get_rank()
            num_replicas = self.dist.get_world_size()
            num_samples = math.ceil(len(wsis) / num_replicas)
            subsample = wsis[rank * num_samples: (rank + 1) * num_samples]


            # make sure each gpu acclocated the same number of wsis
            if len(subsample) < num_samples:
                diff = num_samples - len(subsample)
                subsample.extend(wsis[:diff])

            return subsample


    def get_wsis(self, data_set):
        NotImplementedError

    def set_direction(self, wsis, direction):
        for wsi in wsis:
            wsi.direction = direction

        return wsis

    def set_random_direction(self, wsis):
        random.seed(self.seed)
        for wsi in wsis:
            direction = random.randint(0, 7)
            wsi.direction = direction
        return wsis

    def orgnize_wsis(self, orig_wsis):
        wsis = []
        # when batch size is larger than the total num of wsis
        if self.batch_size > len(orig_wsis):
            if self.allow_reapt:
                wsis = orig_wsis
                while len(wsis) != self.batch_size:
                    wsis.extend(random.sample(orig_wsis, k=1))
            else:
                raise ValueError('allow_reapt should be True when batch_size is larger than the whole wsis')

        else:
            remainder = len(orig_wsis) % self.batch_size
            # if the total number of wsis is not divisible by batch_size
            if remainder > 0:
                # if we do not drop the last, we randomly select "self.batch_size - remainder" number of
                # samples add to the orig_
                random.seed(self.seed)
                if not self.drop_last:
                    wsis = orig_wsis
                    # for wsi in random.sample(self.orig_wsis, k=self.batch_size - remainder):
                    for wsi in random.sample(orig_wsis, k=self.batch_size - remainder):
                        wsis.append(wsi)

                else:
                    # if drop last, we randomly sample "total number of self.orig_wsis - remainer" number
                    # of wsis
                    wsis = random.sample(orig_wsis, k=len(orig_wsis) - remainder)
                    assert len(wsis) == len(orig_wsis) - remainder
            else:
                wsis = orig_wsis
        return wsis


    def shuffle(self, wsis):
        """manually shuffle all the wsis, because when num_workers > 0,
        the copy of dataset wont update in the main process """
        random.seed(self.seed)
        random.shuffle(wsis)

        return wsis

    def cal_seq_len(self, wsis):
        outputs = []

        for idx in range(0, len(wsis), self.batch_size):

            batch_wsi = wsis[idx : idx + self.batch_size]
            max_len = max([wsi.num_patches for wsi in batch_wsi])

            outputs.append(max_len)

        return outputs

    def cycle(self, iterable):
        while True:
            for data in iterable:
                yield data

    def read_img(self, data):

        patch_id = data['patch_id']
        with self.env.begin(write=False) as txn:
            img_stream = txn.get(patch_id.encode())
            img = np.frombuffer(img_stream, np.uint8)
            img = cv2.imdecode(img, -1)  # most time is consum

        data['img'] = img
        return data

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        # create a new list to avoid change the self.orig_wsis
        # during each epoch
        wsis = []
        for wsi in self.orig_wsis:
            wsis.append(wsi)

        wsis = self.shuffle(wsis)
        wsis = self.split_wsis(wsis) # used for ddp training
        wsis = self.orgnize_wsis(wsis)
        global_seq_len = self.cal_seq_len(wsis)

        if self.data_set != 'train':
            wsis = self.set_direction(wsis, direction=0)
            if self.drop_last:
                raise ValueError('during inference, the drop_last should not be set to true')
        else:
            wsis = self.set_random_direction(wsis)

        count = 0
        for idx in range(0, len(wsis), self.batch_size):#0

            # add seeds here to avoid different seed value for
            # different workers if we set seed += 1024 at the
            # end of each data = next(x) loop (count might not
            # be divided by each )
            self.seed += 1024

            batch_wsi = wsis[idx : idx + self.batch_size]

            assert len(batch_wsi) == self.batch_size

            batch_wsi = [self.cycle(x) for x in batch_wsi]

            max_len_idx = idx // self.batch_size

            if not global_seq_len[max_len_idx]:
                warnings.warn('max batch len equals 0')
                continue

            max_len = global_seq_len[max_len_idx]

            # if wsi len is not divisible by num_workers,
            # the last few elements will
            # change the order of reading next round
            # set a global counter to eliminate this issue
            for patch_idx in range(max_len):#104

                    outputs = []
                    for x in batch_wsi:

                        data = next(x)

                        if worker_info is not None:
                            if count % worker_info.num_workers != worker_info.id:
                                continue

                            #data['worker_id'] = worker_info.id
                            #data['count'] = self.count
                            #data['patch_idx'] = patch_idx
                            #data['seed'] = tmp_seed
                            # data['dir'] = tmp_dircts
                        data = self.read_img(data)

                        if patch_idx < max_len - 1:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        outputs.append(data)

                        if self.trans is not None:
                            data['img'] = self.trans(image=data['img'])['image'] # A

                    count += 1

                    if outputs:
                        yield default_collate(outputs)


class CAMLON16Dataset(WSIDatasetNaive, CAMLON16MixIn):
    def __init__(self, data_set, lmdb_path, batch_size, drop_last=False, allow_reapt=False, transforms=None, dist=None, all=True):
        self.all = all
        super().__init__(data_set, lmdb_path, batch_size, drop_last, allow_reapt, transforms, dist)
        # print(self.all, 'cccccccccccccccccccccccccc')

    def get_wsis(self, data_set):
        wsis = self.camlon16_wsis(data_set)
        if not self.all:
            print('hello')
            for wsi in wsis:
                wsi.patch_level()

        return wsis