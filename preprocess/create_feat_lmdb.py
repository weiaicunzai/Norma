import argparse
import random
import math
import os
import io
from itertools import cycle
import csv
import struct

import torch
# from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.utils.data import Dataset, DataLoader
import time
import multiprocessing as mp

from multiprocessing import Process, Queue, Pool, Manager


import torch.distributed as dist
import numpy as np
import cv2
import lmdb

import os
import sys
import struct
sys.path.append(os.getcwd())
import torch
from torchvision import transforms
from PIL import Image
# from model.vit import  get_vit256# 导入 vit_small 函数
from functools import partial
import torch.nn as nn
# from dataset.dataloader_simple import  CAMLON16Dataset
# from dataset.creat_feat_lmdb import LMDBDataset
# from dataset.wsi_reader import camlon16_wsis
import  lmdb
import json
import albumentations as A
import albumentations.pytorch as AP
# import torch.distributed as dist
from preprocess.utils import get_vit256



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

def worker(json_path):
    json_data = json.load(open(json_path, 'r'))
    coords = json_data['coords'][0]

    base_name = json_data['filename']
    keys = []
    t1 = time.time()
    for coord in coords:
        (x, y), level, (patch_size_x, patch_size_y) = coord

        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(level, int)
        assert isinstance(patch_size_x, int)
        assert isinstance(patch_size_y, int)

        patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
            basename=base_name,
            x=x,
            y=y,
            level=level,
            patch_size_x=patch_size_x,
            patch_size_y=patch_size_y)

        keys.append(patch_id)
        # keys.append(patch_id)

    # print(time.time() - t1, len(keys))
    return keys


class PatchLMDB(Dataset):
    def __init__(self, settings, trans) -> None:
        # self.wsis = wsis
        # self.lmdb_path = settings.patch_dir
        self.env = lmdb.open(settings.patch_dir, readonly=True, lock=False)
        print('loading keys .....')
        # with self.env.begin(write=False) as txn:
        self.keys = self.get_keys(settings)
        num_keys = self.env.stat()['entries']
        print(num_keys, len(self.keys))
        assert num_keys == len(self.keys)
        # self.patches = self.get_patch_id(wsis)
        print('done, total {} number of keys'.format(len(self.keys)))
        self.trans = trans
        self.patch_size = settings.patch_size


    # def worker(self, json_path):
    #     json_data = json.load(open(json_path, 'r'))
    #     coords = json_data['coords'][0]

    #     base_name = json_data['filename']
    #     keys = []
    #     t1 = time.time()
    #     for coord in coords:
    #         (x, y), level, (patch_size_x, patch_size_y) = coord

    #         assert isinstance(x, int)
    #         assert isinstance(y, int)
    #         assert isinstance(level, int)
    #         assert isinstance(patch_size_x, int)
    #         assert isinstance(patch_size_y, int)

    #         patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
    #             basename=base_name,
    #             x=x,
    #             y=y,
    #             level=level,
    #             patch_size_x=patch_size_x,
    #             patch_size_y=patch_size_y)

    #         keys.append(patch_id)
    #         # keys.append(patch_id)

    #     # print(time.time() - t1, len(keys))
    #     return keys

        # q.put(keys)

    #def worker(self, json_path, q):
    #    json_data = json.load(open(json_path, 'r'))
    #    coords = json_data['coords'][0]

    #    base_name = json_data['filename']
    #    keys = []
    #    t1 = time.time()
    #    for coord in coords:
    #        (x, y), level, (patch_size_x, patch_size_y) = coord

    #        assert isinstance(x, int)
    #        assert isinstance(y, int)
    #        assert isinstance(level, int)
    #        assert isinstance(patch_size_x, int)
    #        assert isinstance(patch_size_y, int)

    #        patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
    #            basename=base_name,
    #            x=x,
    #            y=y,
    #            level=level,
    #            patch_size_x=patch_size_x,
    #            patch_size_y=patch_size_y)

    #        keys.append(patch_id)
    #        # keys.append(patch_id)

    #    # print(time.time() - t1, len(keys))

    #    q.put(keys)

    #    # print(time.time() - t1, len(keys))

    def get_keys(self, settings):
        csv_path = settings.file_list_csv
        json_dir = settings.json_dir
        json_names = []
        with open(csv_path, 'r') as csv_file:
            for row in csv.DictReader(csv_file):
                slide_id = row['slide_id']
                json_name = os.path.splitext(slide_id)[0] + '.json'
                json_path = os.path.join(json_dir, json_name)
                json_names.append(json_path)

        # mp.set_start_method('fork')
        t1 = time.time()
        pool = Pool(processes=mp.cpu_count())
        keys = pool.map(worker, json_names)
        print('using {:4f}s to load'.format(time.time() - t1))
        # m = Manager()
       # q = Queue()
       # # for i in range(10):
       # #     Process(target=writer, args=(i,q,)).start()
       # # p = Pool(16)

       # proc = []
       # # for path in get_file_path(settings):
       # keys = []
       # num_process = 64
       # count = 0
       # t1 = time.time()
       # for json_name in json_names:
       #     proc.append(
       #         Process(target=self.worker, args=(json_name, q))
       #     )

       #     if len(proc) == num_process:
       #         # print('loading {} slides'.format(len(proc)))
       #         for p in proc:
       #             p.start()


       #         # time.sleep(20)
       #         # print('done')
       #         # time.sleep(10)
       #         while True:
       #             try:
       #                 # if after 10 seconds, still no data
       #                 # then means the process ends
       #                 keys.append(q.get(timeout=2))
       #                 # txn.put(*record)
       #                 count += 1
       #                 # if count % 1000 == 0:
       #                 # print('time', (time.time() - t1) / count, 'total', count)
       #             except Exception as e:
       #                 # print(e)
       #                 # print('end of reading {} processes'.format(len(proc)))
       #                 break


       #         # print('dddddd')
       #         print(count)
       #         # wait untill process ends
       #         for p in proc:
       #             p.join()

       #         # clear prc
       #         proc = []

       # # if last p less len(proc)
       # # print('loading {} slides'.format(len(proc)))
       # for p in proc:
       #     p.start()


       # # print('done')
       # # time.sleep(10)
       # while True:
       #     try:
       #         # if after 10 seconds, still no data
       #         # then means the process ends
       #         keys.append(q.get(timeout=2))
       #         count += 1
       #         # txn.put(*record)
       #         # if count % 1000 == 0:
       #         # print('time', (time.time() - t1) / count, 'total', count)
       #     except:
       #         # print('end of reading {} processes'.format(len(proc)))
       #         break


       # # for json_name in json_names:
       #     # readers.append(p.apply_async(self.worker, (json_name, q)))

       # print(count)
       # for p in proc:
       #     p.join()

       # # res = [r.get() for r in readers]
       # proc = []


        res = []
        for k in keys:
            # print(len(k))
            res.extend(k)


        return res












        # keys = []
        # for json_name in json_names:
        #     json_data = json.load(open(json_name, 'r'))
        #     coords = json_data['coords'][0]

        #     base_name = json_data['filename']
        #     for coord in coords:
        #         (x, y), level, (patch_size_x, patch_size_y) = coord

        #         assert isinstance(x, int)
        #         assert isinstance(y, int)
        #         assert isinstance(level, int)
        #         assert isinstance(patch_size_x, int)
        #         assert isinstance(patch_size_y, int)

        #         patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
        #             basename=base_name,
        #             x=x,
        #             y=y,
        #             level=level,
        #             patch_size_x=patch_size_x,
        #             patch_size_y=patch_size_y)

        #         keys.append(patch_id)

        #     print(len(keys))

        return keys






    def __len__(self):
        return len(self.keys)

    # def get_patch_id(self, wsis):
    #     tmp = []
    #     for wsi in wsis:

    #         for data in wsi:
    #             tmp.append(data['patch_id'])

    #     return tmp

    # def read_img(self, patch_id):
    #     raise NotImplementedError


    def __getitem__(self, idx):
        patch_id = self.keys[idx]
        # print(self.read_img)
        # img = self.read_img(patch_id)
        # print(type(img))

        with self.env.begin(write=False) as txn:
            # print(patch_id)
            img_stream = txn.get(patch_id.encode())
            # mod = index % 9999
            # img_stream = txn.get('cat{:05d}.jpeg'.format(mod).encode())
            # img = Image.open(io.BytesIO(img_stream))
            # img = np.array(img)
            # img = np.frombuffer(img_stream, np.uint8)
            # img = cv2.imdecode(img, -1)
            img = Image.open(io.BytesIO(img_stream))


        # if img.size != (self.patch_size, self.patch_size):
        #     # print(img.size)
        #     img = img.resize((self.patch_size, self.patch_size))

        if self.trans is not None:
            img = self.trans(img)


        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return {'img':img, 'patch_id':patch_id}


        # img['img'] = self.trans(image=img['img'])['image'] # A

        # return img


#class LMDBDataset(Dataset):
#    def __init__(self, lmdb_save_path, wsis, trans, lmdb_read_path) -> None:
#        super().__init__(lmdb_save_path, wsis, trans)
#        # self.lmdb_save_path = lmdb_read_path
#        self.env = lmdb.open(lmdb_read_path, readonly=True, lock=False)
#
#    def read_img(self, patch_id):
#        # print(patch_id, 'cccccccccccc')
#        with self.env.begin(write=False) as txn:
#            # print(patch_id)
#            img_stream = txn.get(patch_id)
#            # mod = index % 9999
#            # img_stream = txn.get('cat{:05d}.jpeg'.format(mod).encode())
#            # img = Image.open(io.BytesIO(img_stream))
#            # img = np.array(img)
#            img = np.frombuffer(img_stream, np.uint8)
#            img = cv2.imdecode(img, -1)
#
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#        return {'img':img, 'patch_id':patch_id.decode()}


# from conf.import settings
# from dataset.wsi_reader import camlon16_wsis
# wsis = camlon16_wsis('train')
# dataset = LMDBDataset(
#     lmdb_save_path='/data/ssd1/by/CAMELYON16/training_lmdb/'
#     wsis=wsis,
#     trans=
# )


# def load_model()


# 1. 实例化模型
#def Scale_VIT(data_set,lmdb_dataset_path,batch_size,lmdb_save):
#    # model = vit_small()
#
#    # 2. 加载预训练权重
#    # pretrained_weights = torch.load('vit256_small_dino.pth')
#    # print(pretrained_weights)
#    # model.load_state_dict(pretrained_weights)
#    model = get_vit256('vit256_small_dino.pth').cuda()
#
#    trans = A.Compose(
#        [
#            A.Resize(256, 256),
#            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#            AP.transforms.ToTensorV2(),
#        ]
#    )
#
#    # 准备数据集
#    # dataset = CAMLON16Dataset(data_set, lmdb_dataset_path, batch_size, transforms=trans, drop_last=False, dist=dist)  # 使用正确的参数初始化你的数据集 cam
#    wsis = camlon16_wsis(data_set)
#    dataset = LMDBDataset(lmdb_read_path=lmdb_dataset_path, lmdb_save_path=lmdb_save, wsis=wsis, trans=trans)
#    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=128)
#
#    # 创建 LMDB 环境
#    env = lmdb.open(lmdb_save, map_size=int(1e12))
#
#
#
#    import time
#    # 提取特征并保存
#    model.eval()
#    with torch.no_grad():
#        count = 0
#        t1 = time.time()
#        # for data in dataset:
#        for data in dataloader:
#            # 读取图像 patch 和 patch_id
#            img, patch_id = data['img'], data['patch_id']
#
#            # print(patch_id)
#            # 将图像 patch 转为 tensor 并通过 ViT 模型提取特征
#            # img_tensor = torch.from_numpy(img).unsqueeze(0)  # 增加批次维度
#            # # ature = model(img_tensor)
#            feature = model(img.cuda())
#
#            count += img.shape[0]
#
#            print(count / (time.time() - t1))
#
#            # 取出 cls_token 作为特征
#            for feat, p_id in zip(feature, patch_id):
#                # cls_token = feature[:, 0].squeeze()
#
#                # print()
#                #将特征转换为 byte string
#                # feature_str = json.dumps(feat.cpu().tolist()).encode()
#                # print(feat.shape)
#                feat = feat.cpu().tolist()
#                # stuck
#                buffer = struct.pack('384f', *feat)
#                print(len(buffer))
#
#                # 保存到 LMDB
#                # with env.begin(write=True) as txn:
#                #    txn.put(p_id.encode(), buffer)

    # env.close()

# from conf.camlon16 import settings

# Scale_VIT(
#     # data_set='val',
#     data_set='train',
#     # lmdb_dataset_path=settings.test_dirs['lmdb'][0],
#     lmdb_dataset_path=settings.train_dirs['lmdb'][0],
#     batch_size=256,
#     lmdb_save='/data/ssd1/by/CAMELYON16/training_feat'
#     # lmdb_save='/data/ssd1/by/CAMELYON16/testing_feat'
# )
def eval_transforms(patch_size):
	"""
	"""
	mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
	eval_t = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ])
	return eval_t


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)
    parser.add_argument('--ckpt', required=True, default=None)

    return parser.parse_args()

# def writer_process(settings, num_patches, q):
def writer_process(settings, q):

    count = 0
    t1 = time.time()

    db_size = 1 << 42
    env = lmdb.open(settings.feat_dir, map_size=db_size)
    # with env.begin() as txn:
        # num_patches = txn.stat()['entries']

    with env.begin(write=True) as txn:
        num_patches = txn.stat()['entries']
        while True:
            record = q.get()

            for patch_id, feat in zip(*record):
                # struct use 2 times less mem storage than torch.save
                # and 3 times faster to decode (struct.unpack+torch.tensor)
                # than torch.load from byte string

                feat = struct.pack('384f', *feat.tolist())
                txn.put(patch_id.encode(), feat)
                count += 1

                if count % 1000 == 0:
                    print('processed {} patches, avg time {:04f}'.format(
                        count,
                        (time.time() - t1) / count
                    ))

            if count == num_patches:
                break

if __name__ == '__main__':
    args = get_args_parser()
    if args.dataset == 'brac':
        from conf.brac import settings

    feat_dir = settings.feat_dir
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    trans = eval_transforms(settings.patch_size)

    # patch_dir = settings.patch_dir

    dataset = PatchLMDB(settings, trans=trans)
    # import sys; sys.exit()
    dataloader = DataLoader(dataset, num_workers=4, batch_size=256 * 4, pin_memory=True, prefetch_factor=8)

    model = get_vit256(args.ckpt).cuda()


    # env =
    # db_size = 1 << 42
    # env = lmdb.open(settings.feat_dir, map_size=db_size)
    # with env.begin() as txn:
        # num_patches = txn.stat()['entries']
    q = Queue()

    # since lmdb only allows one process to write at the same time
    # we use another writer process to perfom writing operation
    # when dataloader is reading data.
    writer_proc = Process(target=writer_process, args=(settings, q))
    writer_proc.start()

    for data in dataloader:
        # print(data)
        # print(data['img'].shape)
        img = data['img'].cuda(non_blocking=True)
        # print(img.shape)

        with torch.no_grad():
            out = model(img)

        out = out.cpu()
        q.put((data['patch_id'], out))

        #for patch_id, feat in zip(data['patch_id'], out):
        #    # struct use 2 times less mem storage than torch.save
        #    # and 3 times faster to decode (struct.unpack+torch.tensor)
        #    # than torch.load from byte string

        #    feat = struct.pack('384f', *feat.tolist())
        #    with env.begin(write=True) as txn:
        #        txn.put(patch_id.encode(), feat)

            # writer_proc

    writer_proc.join()
    print('done processing...')
