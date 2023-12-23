import random
import math
import os
from itertools import cycle

import torch
# from torch.utils.data import Dataset, IterableDataset, DataLoader
import time

import torch.distributed as dist
import numpy as np
import cv2
import lmdb



# class ValDataloader(IterableDataset):

# env = lmdb.open
# lmdb_path = '/data/ssd1/by/CAMELYON16/training_lmdb/'
# env = lmdb.open(lmdb_path, readonly=True, lock=False)

# class LMDBReader:
#     def read_img(self, patch_id):
#         print(patch_id, 'cccccccccccc')
#         with self.env.begin(write=False) as txn:
#             print(patch_id)
#             img_stream = txn.get(patch_id.encode())
#             # mod = index % 9999
#             # img_stream = txn.get('cat{:05d}.jpeg'.format(mod).encode())
#             # img = Image.open(io.BytesIO(img_stream))
#             # img = np.array(img)
#             img = np.frombuffer(img_stream, np.uint8)
#             img = cv2.imdecode(img, -1)

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         return img


class Dataset:
    def __init__(self, lmdb_save_path, wsis, trans) -> None:
        # self.wsis = wsis
        self.lmdb_save_path = lmdb_save_path
        self.patches = self.get_patch_id(wsis)
        self.trans = trans



    def __len__(self):
        return len(self.patches)

    def get_patch_id(self, wsis):
        tmp = []
        for wsi in wsis:

            for data in wsi:
                tmp.append(data['patch_id'])

        return tmp

    def read_img(self, patch_id):
        raise NotImplementedError


    def __getitem__(self, idx):
        patch_id = self.patches[idx]
        # print(self.read_img)
        img = self.read_img(patch_id)
        # print(type(img))

        img['img'] = self.trans(image=img['img'])['image'] # A

        return img


class LMDBDataset(Dataset):
    def __init__(self, lmdb_save_path, wsis, trans, lmdb_read_path) -> None:
        super().__init__(lmdb_save_path, wsis, trans)
        # self.lmdb_save_path = lmdb_read_path
        self.env = lmdb.open(lmdb_read_path, readonly=True, lock=False)

    def read_img(self, patch_id):
        # print(patch_id, 'cccccccccccc')
        with self.env.begin(write=False) as txn:
            # print(patch_id)
            img_stream = txn.get(patch_id.encode())
            # mod = index % 9999
            # img_stream = txn.get('cat{:05d}.jpeg'.format(mod).encode())
            # img = Image.open(io.BytesIO(img_stream))
            # img = np.array(img)
            img = np.frombuffer(img_stream, np.uint8)
            img = cv2.imdecode(img, -1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return {'img':img, 'patch_id':patch_id}


# from conf.import settings
# from dataset.wsi_reader import camlon16_wsis
# wsis = camlon16_wsis('train')
# dataset = LMDBDataset(
#     lmdb_save_path='/data/ssd1/by/CAMELYON16/training_lmdb/'
#     wsis=wsis,
#     trans=
# )
