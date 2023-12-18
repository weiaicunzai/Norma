import os
import sys
sys.path.append(os.getcwd())

# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import logging
import time
import random
import torch


from conf.camlon16 import train_dirs
from dataset.dataloader_simple import CAMLON16Dataset, WSIDatasetNaive


lmdb_path = train_dirs['lmdb'][0]
print(lmdb_path)

# class TestWSI:


class WSITest:
    def __init__(self, start, end):

        """at_mag: extract patches at magnification xxx"""

        # self.env = env


        # parsed_json = json.loads(json_data)
        # parsed_json = self.read_json(patch_json_dir, json_path)
        # self.wsi_label = parsed_json['label']

        # all patches
        # self.coords = parsed_json['coords']

        # only wsi-level patches
        # self.coords = self.parse_coords(parsed_json, json_path)

        # self.num_patches = len(self.coords[0])
        # self.parsed_json = parsed_json

        # assert direction in [0, 1, 2, 3, 4, 5, 6, 7, -1]
        self.data = range(start, end)
        self.direction = -1
        self.num_patches = len(self.data)

    def __str__(self):
        return str(self.data)



    def __iter__(self):

        for idx, i in enumerate(self.data):
            p_label = 0
            if idx == len(self.data) - 1:
                p_label == 1

            yield {
                #'img': self.wsi.read_region(
                #    *coord
                #).convert('RGB'),
                # 'img': img,
                'img': i,
                'label': 0,
                'p_label': 0,
                # tmp
                'patch_id': p_label,

            }


class TestMixIn:
    def dummy_wsis(self, data_set, direction=-1):

        wsis = []
        for i in range(10):
            wsis.append(
                WSITest(start=random.randint(1, 10),  end=random.randint(10, 20))
                # WSITest(start=i * 10,  end=(i+1) * 10)
            )
        for i in wsis:
            print(i.data, end=' ')
        print()
        print(len(wsis))
        return wsis

class Dummy(WSIDatasetNaive, TestMixIn):
    def get_wsis(self, data_set):
        return self.dummy_wsis(data_set)

# ds = CAMLON16Dataset(
#     'train',
#     lmdb_path=lmdb_path,
#     batch_size=16,
#     allow_reapt=True,
#     drop_last=False
# )
ds_dummy = Dummy(
        'train',
    lmdb_path=lmdb_path,
    batch_size=4,
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

# import torch

# class Test(torch.utils.data.IterableDataset):
#     def __init__(self):
#         self.data = range(1000)

#     def __iter__(self):
#         for i in self.data:
#             yield i

# a = Test()

# class SAMP:
# count=0
dataloader = DataLoader(ds_dummy, batch_size=None, shuffle=False, num_workers=4)

for data  in dataloader:
    print(data['img'], '\t\t', data['worker_id'], '\t\t', data['is_last'], '\t', data['count'])
    # for i in data:
    #     print(i['img'], end=' ')
    # count=count+1
    # print("count",count)
    # print()
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            persistent_workers=False
#            )
# # for c in a:
#     # print(c)

# for data in dataloader:
#     print(data)