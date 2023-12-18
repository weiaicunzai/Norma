import random
import warnings
# from itertools import cycle
# from utils.mics import cycle
from torch.utils.data import IterableDataset
import torch
import cv2
from torch.utils.data import default_collate
# default_collate

from PIL import Image
import lmdb
import numpy as np

import time
# from .wsi import WSILMDB
# from conf import camlon16

from .wsi_reader import CAMLON16MixIn
class WSIDatasetNaive(IterableDataset):
    def __init__(self, data_set, lmdb_path, batch_size, drop_last=False, allow_reapt=False, transforms=None):
        """the num_worker of each CAMLON16 dataset is one, """
        assert data_set in ['train', 'val']


        self.batch_size = batch_size
        self.drop_last = drop_last
        self.allow_reapt = allow_reapt
        self.data_set = data_set
        self.trans = transforms

        self.orig_wsis = self.get_wsis(data_set=data_set)
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.seed = 1024

    def get_wsis(self, data_set):
        NotImplementedError

    def set_direction(self, direction):
        for wsi in self.wsis:
            wsi.direction = direction

    def set_random_direction(self):
        random.seed(self.seed)
        for wsi in self.wsis:
            direction = random.randint(0, 7)
            wsi.direction = direction

    def orgnize_wsis(self, wsis):
        wsis = []
        # when batch size is larger than the total num of wsis
        if self.batch_size > len(self.orig_wsis):
            if self.allow_reapt:
                for wsi in self.cycle(self.orig_wsis):
                    wsis.append(wsi)
                    if len(wsis) == self.batch_size:
                        break
            else:
                raise ValueError('allow_reapt should be True')

        else:
            # if
            remainder = len(self.orig_wsis) % self.batch_size
            # if the total number of wsis is not divisible by batch_size
            if remainder > 0:
                # if we do not drop the last, we randomly select "self.batch_size - remainder" number of
                # samples add to the orig_
                random.seed(self.seed)
                if not self.drop_last:
                    wsis = self.orig_wsis
                    for wsi in random.sample(self.orig_wsis, k=self.batch_size - remainder):
                        wsis.append(wsi)

                else:
                    # if drop last, we randomly sample "total number of self.orig_wsis - remainer" number
                    # of wsis
                    wsis = random.sample(self.orig_wsis, k=len(self.orig_wsis) - remainder)

        return wsis





    def shuffle(self):
        """manually shuffle all the wsis, because when num_workers > 0,
        the copy of dataset wont update in the main process """
        random.seed(self.seed)
        random.shuffle(self.wsis)

    def cal_seq_len(self):
        outputs = []

        # print(len(self.wsis), self.batch_size, 'global_seq_len')
        for idx in range(0, len(self.wsis), self.batch_size):
        # for idx in range(0, len(wsis), self.batch_size):

            batch_wsi = self.wsis[idx : idx + self.batch_size]
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
            # img = Image.open(io.BytesIO(img_stream))
            img = np.frombuffer(img_stream, np.uint8)
            img = cv2.imdecode(img, -1)  # most time is consum

        data['img'] = img
        return data

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if self.data_set == 'train':
            self.wsis = self.orgnize_wsis(self.orig_wsis)
            self.global_seq_len = self.cal_seq_len()
            self.set_random_direction()

        # if worker_info.id == 0:
        #     for x in self.wsis:
        #         print(x.data, "ok")
        for idx in range(0, len(self.wsis), self.batch_size):#0



            batch_wsi = self.wsis[idx : idx + self.batch_size]

            # if worker_info.id == 0:
            #     for x in self.wsis:
            #         print(x.data,"ok")

            assert len(batch_wsi) == self.batch_size

            batch_wsi = [self.cycle(x) for x in batch_wsi]

            max_len_idx = idx // self.batch_size

            if not self.global_seq_len[max_len_idx]:
                warnings.warn('max batch len equals 0')
                continue

            max_len = self.global_seq_len[max_len_idx]

            for patch_idx in range(max_len):#104

                    outputs = []
                    # if patch_idx % worker_info.num_workers != worker_info.id:
                    #     continue
                    for x in batch_wsi:

                        data = next(x)

                        if worker_info is not None:
                            if patch_idx % worker_info.num_workers != worker_info.id:
                                continue

                        # data = self.read_img(data)
                        # data['img'] = img
                        # print(max_len)
                        if patch_idx < max_len - 1:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        outputs.append(data)

                        if self.trans is not None:
                            data['img'] = self.trans(image=data['img'])['image'] # A

                        self.seed += 1
                    # print("ok")
                    if outputs :
                        yield default_collate(outputs)
                        # yield outputs,worker_info.id


class CAMLON16Dataset(WSIDatasetNaive, CAMLON16MixIn):

    def get_wsis(self, data_set):
        return self.camlon16_wsis(data_set)
