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
from dataset.dataloader_simple import CAMLON16Dataset, WSIDatasetNaive, CAMLON16DatasetFeat
import torch.distributed as dist



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
            # print(i * 10, (i + 1) * 10)
        for i in wsis:
            print(i.data, end=' ')
        print()
        print(len(wsis))
        return wsis

class Dummy(WSIDatasetNaive, TestMixIn):
    def get_wsis(self, data_set):
        return self.dummy_wsis(data_set)
class Dummy1(CAMLON16DatasetFeat, TestMixIn):

    def get_wsis(self, data_set):
        return self.dummy_wsis(data_set)

    def read_img(self, data_list):

        data = {}
        is_last = 0
        label = None
        feats = []
        # buffer = io.BytesIO()
        # import time
        # t1 = time.time()
        for data in data_list:

            patch_id = data['patch_id']

            # if data['is_last'] == 1:
                # is_last = 1

            if label is None:
                label = data['label']

            # print(label, data['label'])
            # print(data.keys())
            assert label == data['label']

            # print(patch_id)
            # feature_vector = self.cache.get(patch_id, None)
            # if feature_vector is None:
            #     with self.env.begin(write=False) as txn:
            #        img_stream = txn.get(patch_id.encode())
            #     #    feature_vector_list = struct.unpack('384f', img_stream)
            #     #    print(img_stream)
            #        feature_vector = torch.load(io.BytesIO(img_stream))
            #        self.cache[patch_id] = feature_vector
                #    print(feature_vector_list.shape)
                #    feats.append(img_stream)
            # print(data)
            feats.append(data['img'])

        # with self.env.begin(write=False) as txn:
            # feats = []

        # with self.env.begin(write=False) as txn:
        #     feats = [txn.get(x['patch_id'].encode()) for x in data_list]
        # print(time.time() - t1)
        # feats = [torch.load(io.BytesIO(x)) for x in feats]

        data['is_last'] = is_last
        # data['img'] = torch.tensor(feats)
        # data['img'] = torch.tensor(feats)
        # print()
        # data['img'] = torch.stack(feats, dim=0)
        data['img'] = torch.tensor(feats)
        data['label'] = label
        return data

# ds = CAMLON16Dataset(
#     'train',
#     lmdb_path=lmdb_path,
#     batch_size=16,
#     allow_reapt=True,
#     drop_last=False
# )
print(dist)
# ds_dummy = Dummy(
#         'train',
#     lmdb_path=lmdb_path,
#     batch_size=4,
#     allow_reapt=True,
#     drop_last=True,
#     # drop_last=False,
#     dist=dist,
# )

ds_dummy=Dummy1(
    # data_set='val',
    data_set='train',
    # lmdb_path='/data/ssd1/xuxinan/CAMELYON16/testing_feat',
    # lmdb_path='/data/ssd1/by/CAMELYON16/testing_lmdb/',
    # lmdb_path='/data/ssd1/by/CAMELYON16/testing_feat',
    # lmdb_path='/data/ssd1/by/CAMELYON16/testing_feat1/',
    lmdb_path='/data/ssd1/by/CAMELYON16/training_feat1/',
    batch_size=3,
    seq_len=5,
    dist=dist,
    all=True,
    preload=False,
    max_len=20)
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
# from utils.mics import init_process
# init_process()
dataloader = DataLoader(ds_dummy, batch_size=None, shuffle=False, num_workers=4, persistent_workers=True)
# dataloader = DataLoader(ds_dummy, batch_size=None, shuffle=False, num_workers=0)

rank = 2

for _ in range(3):
    # if dist.get_rank() == rank:
    # print('epoch......', dist.get_rank(), dist.get_world_size())
    # time.sleep(1)
    print('----------------------------')
    for data  in dataloader:
        # print(data['img'], '\t\t', data['worker_id'], '\t\t', data['is_last'], '\t', data['count'],  '\t', data['seed'], '\t', data['dir'])
        # print(data['img'], '\t\t', '\t\t', data['is_last'], '\t', data['worker_id'], data['count'], data['patch_idx'] )
        # if dist.get_rank() == rank:
            # print(data['img'], '\t\t', '\t\t', data['is_last'], dist.get_rank())
            # print(data['img'], '\t\t', '\t\t', data['is_last'])
            # print(data)
            # if data['is_last'].sum() > 0:
            # print(data['img'], '\t', data['worker_id'], data['is_last'], data['max_len'], data['patch_id'], data['seq_len'])
            # print(data)
            print('img', data['img'], '\t', 'worker_id', data['worker_id'], 'is_last', data['is_last'])
            # print(data['img'], data['is_last'])

    # dataloader.dataset.
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