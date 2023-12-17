import random
import warnings
# from itertools import cycle
# from utils.mics import cycle
from torch.utils.data import IterableDataset
import torch
import cv2
# default_collate

from PIL import Image

import time


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
        random.shuffle(self.wsis)

    def cal_seq_len(self):
        outputs = []


        print(len(self.wsis), self.batch_size, 'global_seq_len')
        for idx in range(0, len(self.wsis), self.batch_size):

            batch_wsi = self.wsis[idx : idx + self.batch_size]
            max_len = max([wsi.num_patches for wsi in batch_wsi])
            print()
            outputs.append(max_len)

        return outputs

    def cycle(self, iterable):
     while True:
         for data in iterable:
             yield data


    def __iter__(self):

        for idx in range(0, len(self.wsis), self.batch_size):

            batch_wsi = self.wsis[idx : idx + self.batch_size]

            assert len(batch_wsi) == self.batch_size

            batch_wsi = [self.cycle(x) for x in batch_wsi]

            max_len_idx = idx // self.batch_size
            if not self.global_seq_len[max_len_idx]:
                warnings.warn('max batch len equals 0')
                continue
            max_len = self.global_seq_len[max_len_idx]
            for patch_id in range(max_len):

                    outputs = []

                    for x in batch_wsi:

                        data = next(x)

                        if patch_id < max_len - 1:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        # print(patch_id, max_len - 1, data['is_last'])

                        outputs.append(data)

                        if self.trans is not None:
                            data['img'] = self.trans(image=data['img'])['image'] # A

                    yield outputs
