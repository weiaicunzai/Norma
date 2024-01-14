import lmdb
import glob
import os
import json
import io
from PIL import Image
import random
import csv
import sys
import struct
import lmdb
import pandas as pd
import torch
# import numpy as np
# import cv2
# import pandas as p

from torch.utils.data import Dataset

# from conf.camlon16 import settings
sys.path.append(os.getcwd())

# def remove_ext()

class SlideDataset(Dataset):
    def __init__(self, img_set, settings, fold, padding=False):
        assert img_set in ['train', 'val', 'test']
        # self.data_set = data_set
        slide_id = self.fold_slides(settings, img_set, fold)
        # print(slide_id)
        # print(slide_id)
        # import sys; sys.exit()
        # self.data = self.load_json(settings)
        self.data = self.load_json(slide_id, settings)
        self.env = lmdb.open(settings.feat_dir, readonly=True, lock=False)
        self.padding = padding

        if img_set != 'test':
            self.direction = -1
        else:
            self.direction = 0

        # self.fold = fold
        # self.fold_csv =
        self.cache = {}

        self.num_classes = None

        # self.


    def fold_slides(self, settings, img_set, fold):
        file_pd = pd.read_csv(settings.file_list_csv)
        split_path = os.path.join(settings.split_dir, 'splits_{}.csv'.format(fold))
        splits = pd.read_csv(split_path)

        # print(splits)
        print(file_pd['slide_id'][0])
        file_pd['slide_id'] = file_pd['slide_id'].apply(lambda x : os.path.splitext(x)[0])
        print(file_pd['slide_id'][0])

        # print(splits['train'].shape, 'cccccc')
        mask = file_pd['slide_id'].isin(splits[img_set])
        # print(mask.shape)
        # print(file_pd.shape)
        # file_pd.shape
        # print(splits[img_set][0])
        file_pd = file_pd[mask]
        # print(file_pd.shape)
        # print(file_pd.shape)
        # print(file_pd, 'cccc')
        file_pd.reset_index(drop=True, inplace=True)
        return file_pd['slide_id'].tolist()

        # samples = file_pd




    # def get_slide_id(self, settings):
    #     ids = []
    #     with open(settings.file_list_csv) as f:
    #         for row in csv.DictReader(f):
    #             ids.append(row['slide_id'])

    #     return ids
    def __len__(self):
        return len(self.data)

    def load_json(self, slide_ids, settings):
        # for path in glob.iglob(os.path.join(json_dir, '**', '*.json'),):
        # slide_ids = self.get_slide_id(settings)
        data = []
        for slide_id in slide_ids:
            name = slide_id + '.json'
            json_path = os.path.join(settings.json_dir, name)
            json_data = json.load(open(json_path))
            data.append(json_data)

        return data

    def read_data(self, data):
        # for coord
        wsi_label = int(data['label'])
        if self.direction == -1:
            direction = random.randint(0, 7)
        else:
            direction = self.direction

                    # (x, y), level, (patch_size_x, patch_size_y) = coord
                    # print(x, y, level, patch_size_x, patch_size_y)
        patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'
        filename = data['filename']
        # print(filename)
        # .format(
        #                 basename=base_name,
        #                 x=x,
        #                 y=y,
        #                 level=level,
        #                 patch_size_x=patch_size_x,
        #                 patch_size_y=patch_size_y)
        # patch_id.format(basename=filename)
        # key = "{}_{}".format(filename, direction)
        # wsi_feats = self.cache.get(filename, None)
        # if wsi_feats is None:
        if_print = False
        with self.env.begin() as txn:
            feats = []
            for coord in data['coords'][direction]:
                (x, y), level, (patch_size_x, patch_size_y) = coord
                p_id = patch_id.format(
                        basename=filename,
                        x=x,
                        y=y,
                        level=level,
                        patch_size_x=patch_size_x,
                        patch_size_y=patch_size_y
                )

                try:
                    byte_string = txn.get(p_id.encode())
                    feat = struct.unpack('384f', byte_string)
                except:
                    # print(p_id)
                    feat = [x for x in range(384)]
                    if_print = True
                    # import sys; sys.exit()
                # feat = torch.tensor(feat)
                feats.append(feat)

        # print(filename)
        wsi_feats = torch.tensor(feats)
        if if_print:
            print(filename)
        # self.cache[filename] = wsi_feats
        # else:
            # wsi_feats


        # print(wsi_feats.shape)
        # wsi_feats = torch.stack(feats, dim=0)
        # print(feats.shape)

        return wsi_feats, wsi_label




    def __getitem__(self, index):
        data = self.data[index]

        wsi_feats, wsi_label = self.read_data(data)

        return wsi_feats, wsi_label


# from conf.camlon16 import settings
# # pd.set_option('display.max_rows', None)
# from conf.brac import settings
# dataset = SlideDataset(img_set='train', settings=settings, fold=0, padding=True)

# for feat, label in dataset:
#     print(feat.shape, label)
# # # dataset = SlideDataset(img_set='train', settings=settings, fold=0, padding=True)
# # dataset = SlideDataset(img_set='val', settings=settings, fold=0, padding=True)
# dataset = SlideDataset(img_set='test', settings=settings, fold=0, padding=True)

# # print(len(dataset))
# # import cProfile

# # profiler = cProfile.Profile()
# # profiler.enable()

# # for i in range(10):
# count = []
# for feat, label in dataset:
#     count.append(feat.shape[0])
#     # print(feat.shape, label)
# #         pass

# dataset = SlideDataset(img_set='train', settings=settings, fold=0, padding=True)
# for feat, label in dataset:
#     count.append(feat.shape[0])
# # profiler.disable()
# # profiler.print_stats(sort='cumulative')
# dataset = SlideDataset(img_set='val', settings=settings, fold=0, padding=True)
# for feat, label in dataset:
#     count.append(feat.shape[0])


# import matplotlib.pyplot as plt

# plt.hist(count, bins=30, color='blue', edgecolor='black')

# # Add labels and title
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram Example')

# print(sum(count) / len(count))
# # Show the plot
# plt.savefig('hist.jpg')
# print('avg 5923.002506265664')