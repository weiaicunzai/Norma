import os
import sys
sys.path.append(os.getcwd())

# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import logging
import time


from conf.camlon16 import train_dirs
from dataset.dataloader_simple import CAMLON16Dataset

lmdb_path = train_dirs['lmdb'][0]
print(lmdb_path)

ds = CAMLON16Dataset(
    'train',
    lmdb_path=lmdb_path,
    batch_size=16,
    allow_reapt=True,
    drop_last=False
)
# dataset = CAMLON16Dataset(
#     data_set='train',
#     lmdb_path=lmdb_path,
#     batch_size=4,
# )

# for data in dataset:
#     print(data)

import torch

# class Test(torch.utils.data.IterableDataset):
#     def __init__(self):
#         self.data = range(1000)

#     def __iter__(self):
#         for i in self.data:
#             yield i

# a = Test()

# class SAMP:

dataloader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=4)
for data in dataloader:
    #print(data.keys())
    print(len(data))
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            persistent_workers=False
#            )
# # for c in a:
#     # print(c)

# for data in dataloader:
#     print(data)