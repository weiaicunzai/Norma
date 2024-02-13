import random
import time
import torch
import pandas as pd
from pathlib import Path
import functools
import pickle
import os
import sys
from collections import defaultdict
import multiprocessing as mp
import copy
sys.path.append(os.getcwd())

import numpy as np
import torch.utils.data as data
# from torch.utils.data import dataloader
# from torch.utils.data import IterableDataset, default_collate
import pandas as pd
import math
import lmdb
from struct import unpack
from torch.utils.data import default_collate

from datasets.wsi import WSIJSON




class CamelData1(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        '''ideas'''
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle


        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
            pd.set_option('display.max_rows', None)


            val = self.slide_data.loc[:, 'val'].dropna()
            val_label = self.slide_data.loc[:, 'val_label'].dropna()
            # print(val)


            self.data = pd.concat([self.data, val], axis=0).reset_index(drop=True)
            self.label = pd.concat([self.label, val_label], axis=0).reset_index(drop=True)
            # print(self.data)


            # print(type(self.data.loc[[3, 4]]))
            # print(self.data)
            # self.data.iloc[[0, 1]] = self.data.loc[[3, 4]]
            # print(self.data)

            # import sys; sys.exit()

            # 270

            self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_train.pkl', 'rb'))

        if state == 'val':
            # self.data = self.slide_data.loc[:, 'val'].dropna()
            # self.label = self.slide_data.loc[:, 'val_label'].dropna()

            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

            self.label = self.label[self.label != 'test_114'].reset_index(drop=True)
            self.data = self.data[self.data != 'test_114'].reset_index(drop=True)

            self.label = self.label[self.label != 'test_124'].reset_index(drop=True)
            self.data = self.data[self.data != 'test_124'].reset_index(drop=True)

            self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_test.pkl', 'rb'))

        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

        self.wsi_length = 40000 * 2

    def read_data(self, slide_id):
        #tmp = []
        #for feat in self.feats[slide_id]:
        #    tmp.append(feat['feature'])
        #    label = feat['label']

        try:
            feat = [x['feature'] for x in self.feats[slide_id]]
        except Exception as e:
            print(self.state, 'cccccccccccccccccccccccccccccc')
            raise e
            # import sys; sys.exit()


        # feat = torch.tensor(np.array(tmp))
        feat = torch.tensor(np.array(feat))
        # print(feat.shape)
        return feat

    def pad_seq(self, features):
        fact = self.wsi_length / features.shape[0] + 1
        features = features.repeat(int(fact), 1)
        features = features[:self.wsi_length]
        assert features.shape[0] == self.wsi_length

        return features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        # full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        # features = torch.load(full_path)
        # features = self.feats[slide_id]
        features = self.read_data(slide_id)

        #----> shuffle
        if self.shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        # features = features[:4]

        # print(features[:, 4], self.wsi_length)
        # features = self.pad_seq(features)
        # print(features[:, 4], 'after')


        return features, label, slide_id



class CamelData2(data.IterableDataset):
    """使用dftm的预训练权重"""
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle
        self.state = state

        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
            pd.set_option('display.max_rows', None)


            val = self.slide_data.loc[:, 'val'].dropna()
            val_label = self.slide_data.loc[:, 'val_label'].dropna()
            # print(val)


            self.data = pd.concat([self.data, val], axis=0).reset_index(drop=True)
            self.label = pd.concat([self.label, val_label], axis=0).reset_index(drop=True)
            # print(self.data)


            # print(type(self.data.loc[[3, 4]]))
            # print(self.data)
            # self.data.iloc[[0, 1]] = self.data.loc[[3, 4]]
            # print(self.data)

            # import sys; sys.exit()

            # 270

            self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_train.pkl', 'rb'))

        if state == 'val':
            # self.data = self.slide_data.loc[:, 'val'].dropna()
            # self.label = self.slide_data.loc[:, 'val_label'].dropna()

            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

            self.label = self.label[self.label != 'test_114'].reset_index(drop=True)
            self.data = self.data[self.data != 'test_114'].reset_index(drop=True)

            self.label = self.label[self.label != 'test_124'].reset_index(drop=True)
            self.data = self.data[self.data != 'test_124'].reset_index(drop=True)

            self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_test.pkl', 'rb'))

        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

        # self.wsi_length = 40000 * 2
        self.wsi_length = 512 * 78 * 2
        # self.seq_length = int(20000 / 2)
        # self.seq_length = 10000
        # self.seq_length = 5000
        # self.seq_length = 2000
        self.seq_length = 512 * 2
        # self.seq_length = int(20000 / 4)
        # self.wsi_length = 10

    # def load_slide

    def read_data(self, slide_id):
        #tmp = []
        #for feat in self.feats[slide_id]:
        #    tmp.append(feat['feature'])
        #    label = feat['label']

        # try:
        # except Exception as e:
            # print(self.state, 'cccccccccccccccccccccccccccccc')
            # raise e

        feat = [x['feature'] for x in self.feats[slide_id]]
        feat = torch.tensor(np.array(feat))
        return feat

    def pad_seq(self, features):
        fact = self.wsi_length / features.shape[0] + 1
        # print(features[:, 4], fact)
        features = features.repeat(int(fact), 1)
        # print(features[:, 4], 'after repeat....')
        features = features[:self.wsi_length]
        # print(features[:, 4], 'after indexing....')
        assert features.shape[0] == self.wsi_length

        return features

    # def __len__(self):
    #     return len(self.data)

    def __iter__(self):

        idxes = list(range(len(self.data)))

        if self.state == 'train':
            random.shuffle(idxes)

        for idx in idxes:
            slide_id = self.data[idx]
            label = int(self.label[idx])
            # full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            # features = torch.load(full_path)
            # features = self.feats[slide_id]
            features = self.read_data(slide_id)

            # features = features[:4]
            # print(features[:, 4], self.wsi_length)

            #----> shuffle
            if self.shuffle == True:
                index = [x for x in range(features.shape[0])]
                random.shuffle(index)
                features = features[index]

            features = self.pad_seq(features)

            # yield features, label, slide_id
            # before_features = features.clone()
#            # print('before', before_features[:, 4], features[:, 4], 'after', self.wsi_length)
#
#            # for
#            assert self.wsi_length % self.seq_length == 0
#
            num_chunks = self.wsi_length / self.seq_length
            count = 0
            for chunk in features.chunk(int(num_chunks), dim=0):
                count += 1
                if count == num_chunks:
                    is_last = 1
                else:
                    is_last = 0

#
                print(chunk.shape)
#
                yield chunk, label, slide_id, is_last
#




class CamelData1(data.IterableDataset):
    """our (seq) dino dim 384"""
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle
        self.state = state

        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
            pd.set_option('display.max_rows', None)


            val = self.slide_data.loc[:, 'val'].dropna()
            val_label = self.slide_data.loc[:, 'val_label'].dropna()
            # print(val)


            self.data = pd.concat([self.data, val], axis=0).reset_index(drop=True)
            self.label = pd.concat([self.label, val_label], axis=0).reset_index(drop=True)
            # print(self.data)


            # print(type(self.data.loc[[3, 4]]))
            # print(self.data)
            # self.data.iloc[[0, 1]] = self.data.loc[[3, 4]]
            # print(self.data)

            # import sys; sys.exit()

            # 270

            # self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_train.pkl', 'rb'))

        if state == 'val':
            # self.data = self.slide_data.loc[:, 'val'].dropna()
            # self.label = self.slide_data.loc[:, 'val_label'].dropna()

            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

            self.label = self.label[self.label != 'test_114'].reset_index(drop=True)
            self.data = self.data[self.data != 'test_114'].reset_index(drop=True)

            self.label = self.label[self.label != 'test_124'].reset_index(drop=True)
            self.data = self.data[self.data != 'test_124'].reset_index(drop=True)

            # self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_test.pkl', 'rb'))

        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

        # self.wsi_length = 40000 * 2
        self.wsi_length = 512 * 78 * 2
        # self.seq_length = int(20000 / 2)
        # self.seq_length = 10000
        # self.seq_length = 5000
        # self.seq_length = 2000
        self.seq_length = 512 * 2
        # self.seq_length = int(20000 / 4)
        # self.wsi_length = 10

    def read_data(self, slide_id):
        #tmp = []
        #for feat in self.feats[slide_id]:
        #    tmp.append(feat['feature'])
        #    label = feat['label']
        from conf.camlon16 import settings

        json_path = os.path.join(settings.json_dir, slide_id + '.json')


        wsi =  WSIJSON(
                    json_path=json_path,
                    direction=0
        )
        env = lmdb.open(settings.feat_dir, readonly=True, lock=False)
        # with env.
        with env.begin(write=False) as txn:
            output = []
            for i in wsi:
               patch_id = i['patch_id']
                           # label = 0
               # for d in data:
               patch_id = i['patch_id']
               img_stream = txn.get(patch_id.encode())
               feat = unpack('384f', img_stream)
               output.append(feat)

        feat = torch.tensor(output)
        # print(feat.shape, 'cccccccccc')
        # import sys; sys.exit()




        # try:
        #     feat = [x['feature'] for x in self.feats[slide_id]]
        # except Exception as e:
        #     print(self.state, 'cccccccccccccccccccccccccccccc')
        #     raise e
        #     # import sys; sys.exit()


        # # feat = torch.tensor(np.array(tmp))
        # feat = torch.tensor(np.array(feat))
        # print(feat.shape)
        return feat

    def pad_seq(self, features):
        fact = self.wsi_length / features.shape[0] + 1
        # print(features[:, 4], fact)
        features = features.repeat(int(fact), 1)
        # print(features[:, 4], 'after repeat....')
        features = features[:self.wsi_length]
        # print(features[:, 4], 'after indexing....')
        assert features.shape[0] == self.wsi_length

        return features

    # def __len__(self):
    #     return len(self.data)

    def __iter__(self):

        idxes = list(range(len(self.data)))

        if self.state == 'train':
            random.shuffle(idxes)

        for idx in idxes:
            slide_id = self.data[idx]
            label = int(self.label[idx])
            # full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            # features = torch.load(full_path)
            # features = self.feats[slide_id]
            features = self.read_data(slide_id)

            # features = features[:4]
            # print(features[:, 4], self.wsi_length)

            #----> shuffle
            if self.shuffle == True:
                index = [x for x in range(features.shape[0])]
                random.shuffle(index)
                features = features[index]

            features = self.pad_seq(features)

            # yield features, label, slide_id
            # before_features = features.clone()
#            # print('before', before_features[:, 4], features[:, 4], 'after', self.wsi_length)
#
#            # for
#            assert self.wsi_length % self.seq_length == 0
#
            num_chunks = self.wsi_length / self.seq_length
            count = 0
            for chunk in features.chunk(int(num_chunks), dim=0):
                count += 1
                if count == num_chunks:
                    is_last = 1
                else:
                    is_last = 0

#
                print(chunk.shape)
#
                yield chunk, label, slide_id, is_last
#




class CamelData11(data.IterableDataset):
    """clam resnet1024 ours"""
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle
        self.state = state

        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
            pd.set_option('display.max_rows', None)


            val = self.slide_data.loc[:, 'val'].dropna()
            val_label = self.slide_data.loc[:, 'val_label'].dropna()
            # print(val)


            self.data = pd.concat([self.data, val], axis=0).reset_index(drop=True)
            self.label = pd.concat([self.label, val_label], axis=0).reset_index(drop=True)
            # print(self.data)


            # print(type(self.data.loc[[3, 4]]))
            # print(self.data)
            # self.data.iloc[[0, 1]] = self.data.loc[[3, 4]]
            # print(self.data)

            # import sys; sys.exit()

            # 270

            # self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_train.pkl', 'rb'))

        if state == 'val':
            # self.data = self.slide_data.loc[:, 'val'].dropna()
            # self.label = self.slide_data.loc[:, 'val_label'].dropna()

            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

            # self.label = self.label[self.label != 'test_114'].reset_index(drop=True)
            # self.data = self.data[self.data != 'test_114'].reset_index(drop=True)

            # self.label = self.label[self.label != 'test_124'].reset_index(drop=True)
            # self.data = self.data[self.data != 'test_124'].reset_index(drop=True)

            # self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_test.pkl', 'rb'))

        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

        # self.wsi_length = 40000 * 2
        self.wsi_length = 512 * 78 * 2
        # self.seq_length = int(20000 / 2)
        # self.seq_length = 10000
        # self.seq_length = 5000
        # self.seq_length = 2000
        self.seq_length = 512 * 2
        # self.seq_length = int(20000 / 4)
        # self.wsi_length = 10

    def read_data(self, slide_id):
        pt_file = os.path.join('/data/smb/syh/WSI_cls/cam16_using_clam/feature/pt_files/', slide_id + '.pt')

        feat = torch.load(pt_file)

        return feat

    def pad_seq(self, features):
        fact = self.wsi_length / features.shape[0] + 1
        # print(features[:, 4], fact)
        features = features.repeat(int(fact), 1)
        # print(features[:, 4], 'after repeat....')
        features = features[:self.wsi_length]
        # print(features[:, 4], 'after indexing....')
        assert features.shape[0] == self.wsi_length

        return features

    # def __len__(self):
    #     return len(self.data)

    def __iter__(self):

        idxes = list(range(len(self.data)))

        if self.state == 'train':
            random.shuffle(idxes)

        for idx in idxes:
            slide_id = self.data[idx]
            label = int(self.label[idx])
            # full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            # features = torch.load(full_path)
            # features = self.feats[slide_id]
            features = self.read_data(slide_id)

            # features = features[:4]
            # print(features[:, 4], self.wsi_length)

            #----> shuffle
            if self.shuffle == True:
                index = [x for x in range(features.shape[0])]
                random.shuffle(index)
                features = features[index]

            features = self.pad_seq(features)

            # yield features, label, slide_id
            # before_features = features.clone()
#            # print('before', before_features[:, 4], features[:, 4], 'after', self.wsi_length)
#
#            # for
#            assert self.wsi_length % self.seq_length == 0
#
            num_chunks = self.wsi_length / self.seq_length
            count = 0
            for chunk in features.chunk(int(num_chunks), dim=0):
                count += 1
                if count == num_chunks:
                    is_last = 1
                else:
                    is_last = 0

#
                print(chunk.shape)
#
                yield chunk, label, slide_id, is_last
#




class CamelData111(data.IterableDataset):
    """clam dino 384 ours"""
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle
        self.state = state

        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
            pd.set_option('display.max_rows', None)


            val = self.slide_data.loc[:, 'val'].dropna()
            val_label = self.slide_data.loc[:, 'val_label'].dropna()
            # print(val)


            self.data = pd.concat([self.data, val], axis=0).reset_index(drop=True)
            self.label = pd.concat([self.label, val_label], axis=0).reset_index(drop=True)
            # print(self.data)


            # print(type(self.data.loc[[3, 4]]))
            # print(self.data)
            # self.data.iloc[[0, 1]] = self.data.loc[[3, 4]]
            # print(self.data)

            # import sys; sys.exit()

            # 270

            # self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_train.pkl', 'rb'))

        if state == 'val':
            # self.data = self.slide_data.loc[:, 'val'].dropna()
            # self.label = self.slide_data.loc[:, 'val_label'].dropna()

            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

            # self.label = self.label[self.label != 'test_114'].reset_index(drop=True)
            # self.data = self.data[self.data != 'test_114'].reset_index(drop=True)

            # self.label = self.label[self.label != 'test_124'].reset_index(drop=True)
            # self.data = self.data[self.data != 'test_124'].reset_index(drop=True)

            # self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_test.pkl', 'rb'))

        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

        # self.wsi_length = 40000 * 2
        self.wsi_length = 512 * 78 * 2
        # self.seq_length = int(20000 / 2)
        # self.seq_length = 10000
        # self.seq_length = 5000
        # self.seq_length = 2000
        self.seq_length = 512 * 2
        # self.seq_length = int(20000 / 4)
        # self.wsi_length = 10

    def read_data(self, slide_id):

        path = '/data/smb/syh/WSI_cls/cam16_using_clam_dino384_use_pretrain/feature/pt_files'
        # path = '/data/smb/syh/WSI_cls/cam16_using_clam_dino384_not_use_pretrain/feature/pt_files/'

        pt_file = os.path.join(path, slide_id + '.pt')
        print('reading from {}'.format(path))

        feat = torch.load(pt_file)

        return feat

    def pad_seq(self, features):
        fact = self.wsi_length / features.shape[0] + 1
        # print(features[:, 4], fact)
        features = features.repeat(int(fact), 1)
        # print(features[:, 4], 'after repeat....')
        features = features[:self.wsi_length]
        # print(features[:, 4], 'after indexing....')
        assert features.shape[0] == self.wsi_length

        return features

    # def __len__(self):
    #     return len(self.data)


    def __iter__(self):

        idxes = list(range(len(self.data)))

        if self.state == 'train':
            random.shuffle(idxes)

        for idx in idxes:
            slide_id = self.data[idx]
            label = int(self.label[idx])
            # full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            # features = torch.load(full_path)
            # features = self.feats[slide_id]
            features = self.read_data(slide_id)

            # features = features[:4]
            # print(features[:, 4], self.wsi_length)

            #----> shuffle
            if self.shuffle == True:
                index = [x for x in range(features.shape[0])]
                random.shuffle(index)
                features = features[index]

            features = self.pad_seq(features)

            yield features, label, slide_id

            # yield features, label, slide_id
            # before_features = features.clone()
#            # print('before', before_features[:, 4], features[:, 4], 'after', self.wsi_length)
#
#            # for
#            assert self.wsi_length % self.seq_length == 0
#
            # num_chunks = self.wsi_length / self.seq_length
            # count = 0
#             for chunk in features.chunk(int(num_chunks), dim=0):
#                 count += 1
#                 if count == num_chunks:
#                     is_last = 1
#                 else:
#                     is_last = 0

# #
#                 print(chunk.shape)
# #
#                 yield chunk, label, slide_id, is_last
# #




class CamelData(data.IterableDataset):
    """clam resnet1024 ours"""
    def __init__(self, dataset_cfg=None,
                 state=None, settings=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        # if settings is not None:

        #---->data and label
        # self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle
        self.state = state

        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
            pd.set_option('display.max_rows', None)


            val = self.slide_data.loc[:, 'val'].dropna()
            val_label = self.slide_data.loc[:, 'val_label'].dropna()
            # print(val)


            self.data = pd.concat([self.data, val], axis=0).reset_index(drop=True)
            self.label = pd.concat([self.label, val_label], axis=0).reset_index(drop=True)
            # print(self.data)


            # print(type(self.data.loc[[3, 4]]))
            # print(self.data)
            # self.data.iloc[[0, 1]] = self.data.loc[[3, 4]]
            # print(self.data)

            # import sys; sys.exit()

            # 270

            # self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_train.pkl', 'rb'))

        if state == 'val':
            # self.data = self.slide_data.loc[:, 'val'].dropna()
            # self.label = self.slide_data.loc[:, 'val_label'].dropna()

            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

            # self.label = self.label[self.label != 'test_114'].reset_index(drop=True)
            # self.data = self.data[self.data != 'test_114'].reset_index(drop=True)

            # self.label = self.label[self.label != 'test_124'].reset_index(drop=True)
            # self.data = self.data[self.data != 'test_124'].reset_index(drop=True)

            # self.feats = pickle.load(open('/data/smb/syh/WSI_cls/mDATA_test.pkl', 'rb'))

        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

        # self.wsi_length = 40000 * 2
        self.wsi_length = 512 * 78 * 2
        # self.seq_length = int(20000 / 2)
        # self.seq_length = 10000
        # self.seq_length = 5000
        # self.seq_length = 2000
        self.seq_length = 512 * 2
        # self.seq_length = int(20000 / 4)
        # self.wsi_length = 10

    def read_data(self, slide_id):

        path = '/data/smb/syh/WSI_cls/cam16_using_clam/feature/pt_files/'
        # path = '/data/smb/syh/WSI_cls/cam16_using_clam_dino384_not_use_pretrain/feature/pt_files/'

        pt_file = os.path.join(path, slide_id + '.pt')
        print('reading from {}'.format(path))

        feat = torch.load(pt_file)

        return feat

    def read_data1(self, slide_id):

        # json_path = os.path.join(settings.json_dir, slide_id + '.json')
        # pt_file = os.path.join('/data/smb/syh/WSI_cls/cam16_using_clam/feature/pt_files/', slide_id + '.pt')
        # pt_file = os.path.join('/data/smb/syh/WSI_cls/cam16_using_clam_dino384_not_use_pretrain/feature/pt_files/', slide_id + '.pt')
        # path = '/data/smb/syh/WSI_cls/cam16_using_clam_dino384_not_use_pretrain/feature/pt_files/'
        # path = '/data/smb/syh/WSI_cls/cam16_using_clam_dino384_use_pretrain/feature/pt_files'
        path = '/data/smb/syh/WSI_cls/cam16/feat/patch_size_256_at_mag_20/'
        pt_file = os.path.join(path, slide_id + '.pt')
        # pt_file = os.path.join('/data/smb/syh/WSI_cls/cam16_using_clam_dino384_use_pretrain/feature/pt_files', slide_id + '.pt')
        print('reading from {}'.format(path))

        feats = torch.load(pt_file)
        output = []
        for feat in feats.values():
            output.append(feat)


        # print(feats.shape)
        # import sys; sys.exit()

    #     feat = torch.tensor(output)
        # print(feat.shape, 'cccccccccc')
        # import sys; sys.exit()




        # try:
        #     feat = [x['feature'] for x in self.feats[slide_id]]
        # except Exception as e:
        #     print(self.state, 'cccccccccccccccccccccccccccccc')
        #     raise e
        #     # import sys; sys.exit()


        # # feat = torch.tensor(np.array(tmp))
        # feat = torch.tensor(np.array(feat))
        # print(feat.shape)
        feat = torch.tensor(output)
        # feat = torch.stack(output, dim=0)
        # print(output.shape)
        return feat

    def pad_seq(self, features):
        fact = self.wsi_length / features.shape[0] + 1
        # print(features[:, 4], fact)
        features = features.repeat(int(fact), 1)
        # print(features[:, 4], 'after repeat....')
        features = features[:self.wsi_length]
        # print(features[:, 4], 'after indexing....')
        assert features.shape[0] == self.wsi_length

        return features

    # def __len__(self):
    #     return len(self.data)

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        # create a new list to avoid change the self.orig_wsis
        # during each epoch
        wsis = []
        for wsi in self.orig_wsis:
            wsis.append(wsi)

        wsis = self.shuffle(wsis)
        # wsis = self.split_wsis(wsis) # used for ddp training
        # wsis = self.orgnize_wsis(wsis)
        # global_seq_len = self.cal_seq_len(wsis)

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

            # assert len(batch_wsi) == self.batch_size
            if len(batch_wsi) < self.batch_size:
                if self.drop_last:
                    continue

            batch_wsi = [self.cycle(x) for x in batch_wsi]

            # max_len_idx = idx // self.batch_size
            # mex_len = self.wsi_len

            # if not global_seq_len[max_len_idx]:
                # warnings.warn('max batch len equals 0')
                # continue

            # max_len = global_seq_len[max_len_idx]

            # if wsi len is not divisible by num_workers,
            # the last few elements will
            # change the order of reading next round
            # set a global counter to eliminate this issue
            for patch_idx in range(0, self.wsi_len, self.seq_len):#104

                    outputs = []
                    for wsi in batch_wsi:

                        # data = next(wsi)
                        data = self.read_seq(wsi)

                        if worker_info is not None:
                            if count % worker_info.num_workers != worker_info.id:
                                continue

                        data = self.read_img(data)

                        # if patch_idx < max_len - 1:
                        # if patch_idx < self.wsi_len - 1:
                        if patch_idx < self.wsi_len - self.seq_len:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        outputs.append(data)

                    count += 1

                    if outputs:
                        data = default_collate(outputs)
                        # print(data['label'])
                        yield data['feat'], data['label'], data['filename'], data['is_last']


    def __iter__(self):

        idxes = list(range(len(self.data)))

        if self.state == 'train':
            random.shuffle(idxes)

        for idx in idxes:
            slide_id = self.data[idx]
            label = int(self.label[idx])
            # full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            # features = torch.load(full_path)
            # features = self.feats[slide_id]
            features = self.read_data(slide_id)

            # features = features[:4]
            # print(features[:, 4], self.wsi_length)

            #----> shuffle
            if self.shuffle == True:
                index = [x for x in range(features.shape[0])]
                random.shuffle(index)
                features = features[index]

            features = self.pad_seq(features)

            # yield features, label, slide_id
            # before_features = features.clone()
#            # print('before', before_features[:, 4], features[:, 4], 'after', self.wsi_length)
#
#            # for
#            assert self.wsi_length % self.seq_length == 0
#
            num_chunks = self.wsi_length / self.seq_length
            count = 0
            for chunk in features.chunk(int(num_chunks), dim=0):
                count += 1
                if count == num_chunks:
                    is_last = 1
                else:
                    is_last = 0

#
                # print(chunk.shape)
#
                yield chunk, label, slide_id, is_last



# class WSIDataset(data.IterableDataset):
#     # def __init__(self, data_set, lmdb_path, batch_size, drop_last=False, allow_reapt=False, transforms=None, dist=None):
#     def __init__(self, data_set, fold, lmdb_path, batch_size, drop_last=False, allow_reapt=False, transforms=None, dist=None):
#         """the num_worker of each CAMLON16 dataset is one, """
#         # assert data_set in ['train', 'val']


#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.allow_reapt = allow_reapt
#         self.data_set = data_set
#         self.fold = fold
#         self.trans = transforms

#         # self.orig_wsis = self.get_wsis(data_set=data_set)
#         self.orig_wsis = self.get_wsis(data_set=self.data_set, fold=self.fold)
#         self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
#         self.seed = 52

#         self.dist = dist

#         self.seq_len = 1024
#         self.wsi_len = 79872

#     def split_wsis(self, wsis):
#         if not self.dist.is_initialized():
#             return wsis
#         else:
#             """get wsis for each gpu"""
#             rank = self.dist.get_rank()
#             num_replicas = self.dist.get_world_size()
#             num_samples = math.ceil(len(wsis) / num_replicas)
#             subsample = wsis[rank * num_samples: (rank + 1) * num_samples]

#             # make sure each gpu acclocated the same number of wsis
#             if len(subsample) < num_samples:
#                 diff = num_samples - len(subsample)
#                 subsample.extend(wsis[:diff])

#             return subsample


#     def get_wsis(self, data_set):
#         NotImplementedError

#     def set_direction(self, wsis, direction):
#         for wsi in wsis:
#             wsi.direction = direction

#         return wsis

#     def set_random_direction(self, wsis):
#         random.seed(self.seed)
#         for wsi in wsis:
#             direction = random.randint(0, 7)
#             wsi.direction = direction
#         return wsis

#     def orgnize_wsis(self, orig_wsis):
#         wsis = []
#         # when batch size is larger than the total num of wsis
#         if self.batch_size > len(orig_wsis):
#             if self.allow_reapt:
#                 wsis = orig_wsis
#                 while len(wsis) != self.batch_size:
#                     wsis.extend(random.sample(orig_wsis, k=1))
#             else:
#                 raise ValueError('allow_reapt should be True when batch_size is larger than the whole wsis')

#         else:
#             remainder = len(orig_wsis) % self.batch_size
#             # if the total number of wsis is not divisible by batch_size
#             if remainder > 0:
#                 # if we do not drop the last, we randomly select "self.batch_size - remainder" number of
#                 # samples add to the orig_
#                 random.seed(self.seed)
#                 if not self.drop_last:
#                     wsis = orig_wsis
#                     # for wsi in random.sample(self.orig_wsis, k=self.batch_size - remainder):
#                     for wsi in random.sample(orig_wsis, k=self.batch_size - remainder):
#                         wsis.append(wsi)

#                 else:
#                     # if drop last, we randomly sample "total number of self.orig_wsis - remainer" number
#                     # of wsis
#                     wsis = random.sample(orig_wsis, k=len(orig_wsis) - remainder)
#                     assert len(wsis) == len(orig_wsis) - remainder
#             else:
#                 wsis = orig_wsis
#         return wsis


#     def shuffle(self, wsis):
#         """manually shuffle all the wsis, because when num_workers > 0,
#         the copy of dataset wont update in the main process """
#         random.seed(self.seed)
#         random.shuffle(wsis)

#         return wsis

#     def cal_seq_len(self, wsis):
#         outputs = []

#         for idx in range(0, len(wsis), self.batch_size):

#             batch_wsi = wsis[idx : idx + self.batch_size]
#             max_len = max([wsi.num_patches for wsi in batch_wsi])

#             outputs.append(max_len)

#         return outputs

#     def cycle(self, iterable):
#         while True:
#             for data in iterable:
#                 yield data

#     def read_img(self, data):

#         with self.env.begin(write=False) as txn:
#             output = []
#             label = 0
#             for d in data:
#                 patch_id = data['patch_id']
#                 img_stream = txn.get(patch_id.encode())
#                 feat = unpack('384f', img_stream)
#                 output.append(feat)
#                 label = data['label']

#                 # img = np.frombuffer(img_stream, np.uint8)
#                 # In the case of color images, the decoded images will have the channels stored in B G R order.
#                 # img = cv2.imdecode(img, -1)  # most time is consum
#                 # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#         data['feat'] = torch.tensor(output)
#         data['label'] = label
#         return data

#     def read_seq(self, wsi):
#         outputs = []
#         for i in range(self.seq_len):
#             outputs.append(next(wsi))

#         return outputs

#     def __iter__(self):

#         worker_info = torch.utils.data.get_worker_info()

#         # create a new list to avoid change the self.orig_wsis
#         # during each epoch
#         wsis = []
#         for wsi in self.orig_wsis:
#             wsis.append(wsi)

#         wsis = self.shuffle(wsis)
#         # wsis = self.split_wsis(wsis) # used for ddp training
#         # wsis = self.orgnize_wsis(wsis)
#         # global_seq_len = self.cal_seq_len(wsis)

#         if self.data_set != 'train':
#             wsis = self.set_direction(wsis, direction=0)
#             if self.drop_last:
#                 raise ValueError('during inference, the drop_last should not be set to true')
#         else:
#             wsis = self.set_random_direction(wsis)

#         count = 0
#         for idx in range(0, len(wsis), self.batch_size):#0

#             # add seeds here to avoid different seed value for
#             # different workers if we set seed += 1024 at the
#             # end of each data = next(x) loop (count might not
#             # be divided by each )
#             self.seed += 1024

#             batch_wsi = wsis[idx : idx + self.batch_size]

#             # assert len(batch_wsi) == self.batch_size

#             batch_wsi = [self.cycle(x) for x in batch_wsi]

#             # max_len_idx = idx // self.batch_size
#             # mex_len = self.wsi_len

#             # if not global_seq_len[max_len_idx]:
#                 # warnings.warn('max batch len equals 0')
#                 # continue

#             # max_len = global_seq_len[max_len_idx]

#             # if wsi len is not divisible by num_workers,
#             # the last few elements will
#             # change the order of reading next round
#             # set a global counter to eliminate this issue
#             for patch_idx in range(0, self.wsi_len, self.seq_len):#104

#                     outputs = []
#                     for wsi in batch_wsi:

#                         # data = next(wsi)
#                         data = self.read_seq(wsi)

#                         if worker_info is not None:
#                             if count % worker_info.num_workers != worker_info.id:
#                                 continue

#                             #data['worker_id'] = worker_info.id
#                             #data['count'] = self.count
#                             #data['patch_idx'] = patch_idx
#                             #data['seed'] = tmp_seed
#                             # data['dir'] = tmp_dircts
#                         data = self.read_img(data)

#                         # if patch_idx < max_len - 1:
#                         # if patch_idx < self.wsi_len - 1:
#                         if patch_idx < self.wsi_len - self.seq_len:
#                             data['is_last'] = 0
#                         else:
#                             data['is_last'] = 1

#                         outputs.append(data)

#                     count += 1

#                     if outputs:
#                         yield default_collate(outputs)




# class CAM16(WSIDataset):

def get_wsi(slide, json_dir):
    json_path = os.path.join(json_dir, slide + '.json')
    print('load json {}'.format(json_path))
    return WSIJSON(json_path=json_path, direction=0)

class WSIDataset1(data.IterableDataset):
    # def __init__(self, data_set, lmdb_path, batch_size, drop_last=False, allow_reapt=False, transforms=None, dist=None):
    # def __init__(self, data_set, fold, batch_size, drop_last=False, allow_reapt=False, dist=None):
    def __init__(self, settings, data_set, fold, batch_size, drop_last=False, allow_reapt=False, dist=None, random_shuffle=True):
        """the num_worker of each CAMLON16 dataset is one, """
        # assert data_set in ['train', 'val']
        print(self)


        self.batch_size = batch_size
        self.drop_last = drop_last
        self.allow_reapt = allow_reapt
        self.data_set = data_set
        self.fold = fold
        # self.trans = transforms
        # from conf.camlon16 import settings
        self.settings = settings

        # self.orig_wsis = self.get_wsis(data_set=data_set)
        self.orig_wsis = self.get_wsis(data_set=self.data_set, fold=self.fold)
        print('total number of wsis: {}'.format(len(self.orig_wsis)))
        self.env = lmdb.open(settings.feat_dir, readonly=True, lock=False)
        self.seed = 52

        self.dist = dist

        self.max_len = settings.max_len
        self.seq_len = 1024
        num_chunks = math.ceil(self.max_len / self.seq_len)
        # self.wsi_len = 79872
        self.wsi_len = num_chunks * 2 * self.seq_len
        print('wsi length: {}'.format(self.wsi_len))
        self.random_shuffle = random_shuffle

        if self.data_set == 'train':
            print('shuffle at __init__ val')
            self.orig_wsis = [self.shuffle_coords(x) for x in self.orig_wsis]
        # print('shuffle once val + train')
        # self.orig_wsis = [self.shuffle_coords(x) for x in self.orig_wsis]


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

    def shuffle_coords(self, wsi):
        wsi = copy.deepcopy(wsi)
        for i in range(len(wsi.coords)):
            random.seed(self.seed)
            random.shuffle(wsi.coords[i])

        return wsi

    # def random_shuffle(wsis)

    def get_wsis(self, data_set, fold):
        # wsis = self.data_set()
        # from conf.camlon16 import settings
        file_list = pd.read_csv(self.settings.file_list_csv)
        file_list['slide_id'] = file_list['slide_id'].apply(
            lambda x: os.path.splitext(x)[0]
        )
        # print(file_list['slide_id'])
        # splits = os.path.join(os.pat)
        # split_file = os.path.join('datasets', 'splits', 'cam16', 'splits_{}.csv'.format(fold))
        # pd.set_option('display.max_rows', None)
        split_file = os.path.join(self.settings.split_dir, 'splits_{}.csv'.format(fold))
        splits = pd.read_csv(split_file)
        # data_list = splits[data_set]
        # if data_set != 'test':
        train_split = splits['train'].dropna()
        val_split = splits['val'].dropna()
        test_split = splits['test'].dropna()
        # print(data_list)
        # print(val_split, train_split, test_split)


        if data_set != 'test':
            mask1 = file_list['slide_id'].isin(train_split)
            mask2 = file_list['slide_id'].isin(val_split)
            slide_ids = file_list['slide_id'][mask1 | mask2]
        else:
            mask1 = file_list['slide_id'].isin(test_split)
            slide_ids = file_list['slide_id'][mask1]


        # print(slide_ids)
        wsis = []



        # pool = mp.Pool(processes=mp.cpu_count())
        print('start load json')
        t1 = time.time()
        # pool = mp.Pool(processes=mp.cpu_count())
        # pool = mp.Pool(processes=4)
        # fn = functools.partial(get_wsi, json_dir=self.settings.json_dir)
        # wsis = pool.map(fn, slide_ids.tolist())
        wsis = []
        for slide_id in slide_ids:
            wsi = get_wsi(slide_id, self.settings.json_dir)
            wsis.append(wsi)

        print('done load json {}'.format(time.time() - t1))
        # for slide in slide_ids:
        #     # print(i, '333')
        #     json_path = os.path.join(self.settings.json_dir, slide + '.json')
        #     print(json_path)
        #     wsis.append(
        #         WSIJSON(
        #             json_path=json_path,
        #             direction=0,
        #             )
        #     )


        # mask = file_list.isin(sli)
        # print(len(slide_ids))

        # data = file_list[mask]
        # print(data)
        # return data
        return wsis

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

    def read_seq(self, wsi):

        t1 = time.time()

        feats = []
        label = 0
        filename = ''
        with self.env.begin(write=False) as txn:
            for idx, d in enumerate(wsi):
                patch_id = d['patch_id']

                img_stream = txn.get(patch_id.encode())
                # feat = self.data[patch_id]

                # feat = unpack('384f', img_stream)
                feat = unpack('1024f', img_stream)
                # feat = torch.tensor(feat)
                feats.append(feat)
                label = d['label']
                filename = d['filename']
                # print(label)

                # img = np.frombuffer(img_stream, np.uint8)
                # In the case of color images, the decoded images will have the channels stored in B G R order.
                # img = cv2.imdecode(img, -1)  # most time is consum
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if idx == self.seq_len - 1:
                    break




        output = {}
        # output['feat'] = torch.stack(feats, dim=0)
        t3 = time.time()
        # output['feat'] = torch.tensor(feats)
        # output['feat'] = torch.stack(feats, dim=0)
        output['feat'] = feats
        output['label'] = int(label)
        output['filename'] = filename
        t2 = time.time()
        # print(t2 - t1, t3 - t1, (t3 - t1) / (t2 - t1))

        return output

    def shift_wsi(self, batch_wsi, num_patches):
        random.seed(self.seed)
        # shift 1 or self.seq_len - 1 times
        # shift = random.randint(1,  - 1)
        # print('totoal {} number of wsis , shift {} number of tokens seq_len {} '.format(len(batch_wsi), shift, self.seq_len))
        tmp = []
        for wsi, num_patch in zip(batch_wsi, num_patches):

            shift = random.randint(0, num_patch - 1)
            for i in range(shift):
                next(wsi)

            tmp.append(wsi)

        return tmp

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        # create a new list to avoid change the self.orig_wsis
        # during each epoch
        wsis = []
        for wsi in self.orig_wsis:
            wsis.append(wsi)

        wsis = self.shuffle(wsis)
        # wsis = self.split_wsis(wsis) # used for ddp training
        # wsis = self.orgnize_wsis(wsis)
        # global_seq_len = self.cal_seq_len(wsis)

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

            # if self.data_set == 'train':
            if self.data_set != 'train':
                print('val shuffle each epoch')
                batch_wsi = [self.shuffle_coords(x) for x in batch_wsi]

            # print('shuffle each epoch train+val')
            # batch_wsi = [self.shuffle_coords(x) for x in batch_wsi]

            # assert len(batch_wsi) == self.batch_size
            if len(batch_wsi) < self.batch_size:
                if self.drop_last:
                    continue

            num_patches = [w.num_patches for w in batch_wsi]
            batch_wsi = [self.cycle(x) for x in batch_wsi]

            if self.data_set == 'train':
                batch_wsi = self.shift_wsi(batch_wsi, num_patches)

            # max_len_idx = idx // self.batch_size
            # mex_len = self.wsi_len

            # if not global_seq_len[max_len_idx]:
                # warnings.warn('max batch len equals 0')
                # continue

            # max_len = global_seq_len[max_len_idx]

            # if wsi len is not divisible by num_workers,
            # the last few elements will
            # change the order of reading next round
            # set a global counter to eliminate this issue
            for patch_idx in range(0, self.wsi_len, self.seq_len):#104

                    outputs = []
                    for wsi in batch_wsi:

                        # data = next(wsi)
                        data = self.read_seq(wsi)

                        if worker_info is not None:
                            if count % worker_info.num_workers != worker_info.id:
                                continue

                            #data['worker_id'] = worker_info.id
                            #data['count'] = self.count
                            #data['patch_idx'] = patch_idx
                            #data['seed'] = tmp_seed
                            # data['dir'] = tmp_dircts
                        # data = self.read_img(data)

                        # if patch_idx < max_len - 1:
                        # if patch_idx < self.wsi_len - 1:
                        if patch_idx < self.wsi_len - self.seq_len:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        outputs.append(data)

                    count += 1

                    if outputs:
                        # data = default_collate(outputs)
                        data = dict_collate_fn(outputs, tensor_keys=['feat', 'label', 'is_last'])
                        # print(data['feat'])
                        # data['feat'] = torch.stack(data['feat'], dim=0)
                        # print(type(data['feat']))
                        # for d in data['feat']:
                        #     print(d.shape)
                        # print(data['feat'].shape)
                        # print(data['label'])
                        yield data['feat'], data['label'], data['filename'], data['is_last']



class WSIDataset(data.IterableDataset):

    def __init__(self, settings, data_set, fold, batch_size, drop_last=False, allow_reapt=False, dist=None):
        """the num_worker of each CAMLON16 dataset is one, """
        # assert data_set in ['train', 'val']
        print(self)

        assert data_set in ['train', 'val', 'test']


        self.batch_size = batch_size
        self.drop_last = drop_last
        self.allow_reapt = allow_reapt
        self.data_set = data_set
        self.fold = fold
        # self.trans = transforms
        # from conf.camlon16 import settings
        self.settings = settings

        # self.orig_wsis = self.get_wsis(data_set=data_set)
        self.orig_wsis = self.get_wsis(data_set=self.data_set, fold=self.fold)
        print('total number of wsis: {}'.format(len(self.orig_wsis)))
        self.env = lmdb.open(settings.feat_dir, readonly=True, lock=False)
        self.seed = 52

        self.dist = dist

        self.max_len = settings.max_len
        self.seq_len = 1024
        num_chunks = math.ceil(self.max_len / self.seq_len)
        # self.wsi_len = 79872
        self.wsi_len = num_chunks * 2 * self.seq_len
        print('wsi length: {}'.format(self.wsi_len))

        #print('loading all the data')
        #self.cache = self.load_data()
        #print('done')
        self.cache = {}

    def load_data(self):
        # with self.env.begin(write=False):
        #     cursor = txn.cursor()
        #     for key, _ in cursor:
        #         keys.append(key)

        res = {}
        for idx, wsi in enumerate(self.orig_wsis):
            for data in wsi:
                patch_id = data['patch_id']
                res[patch_id] = self.read_feat(patch_id)

            print(idx)
        #res = {}
        #for key in keys:
        return res

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


    def get_wsis(self, data_set, fold):
        # wsis = self.data_set()
        # from conf.camlon16 import settings
        file_list = pd.read_csv(self.settings.file_list_csv)
        file_list['slide_id'] = file_list['slide_id'].apply(
            lambda x: os.path.splitext(x)[0]
        )
        # print(file_list['slide_id'])
        # splits = os.path.join(os.pat)
        # split_file = os.path.join('datasets', 'splits', 'cam16', 'splits_{}.csv'.format(fold))
        # pd.set_option('display.max_rows', None)
        split_file = os.path.join(self.settings.split_dir, 'splits_{}.csv'.format(fold))
        splits = pd.read_csv(split_file)
        # data_list = splits[data_set]
        # if data_set != 'test':
        train_split = splits['train'].dropna()
        val_split = splits['val'].dropna()
        test_split = splits['test'].dropna()
        # print(data_list)
        # print(val_split, train_split, test_split)


        # if data_set != 'test':
        if data_set == 'train':
            mask1 = file_list['slide_id'].isin(train_split)
            mask2 = file_list['slide_id'].isin(val_split)
            slide_ids = file_list['slide_id'][mask1 | mask2]
        else:
            mask1 = file_list['slide_id'].isin(test_split)
            slide_ids = file_list['slide_id'][mask1]


        # print(slide_ids)
        wsis = []



        # pool = mp.Pool(processes=mp.cpu_count())
        print('start load json')
        t1 = time.time()
        # pool = mp.Pool(processes=mp.cpu_count())
        # pool = mp.Pool(processes=4)
        # fn = functools.partial(get_wsi, json_dir=self.settings.json_dir)
        # wsis = pool.map(fn, slide_ids.tolist())
        wsis = []
        for slide_id in slide_ids:
            wsi = get_wsi(slide_id, self.settings.json_dir)
            wsis.append(wsi)

        print('done load json {}'.format(time.time() - t1))
        # for slide in slide_ids:
        #     # print(i, '333')
        #     json_path = os.path.join(self.settings.json_dir, slide + '.json')
        #     print(json_path)
        #     wsis.append(
        #         WSIJSON(
        #             json_path=json_path,
        #             direction=0,
        #             )
        #     )


        # mask = file_list.isin(sli)
        # print(len(slide_ids))

        # data = file_list[mask]
        # print(data)
        # return data
        return wsis




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

        output = {}
        # with self.env.begin(write=False) as txn:
        feats = []
        label = 0
        for d in data:
            patch_id = d['patch_id']
            # img_stream = txn.get(patch_id.encode())
            feat = self.data[patch_id]

            # feat = unpack('384f', img_stream)
            # feat = unpack('1024f', img_stream)
            feats.append(feat)
            label = d['label']
                # print(label)

                # img = np.frombuffer(img_stream, np.uint8)
                # In the case of color images, the decoded images will have the channels stored in B G R order.
                # img = cv2.imdecode(img, -1)  # most time is consum
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # output['feat'] = torch.tensor(feats)
        # output['feat'] = torch.tensor(feats)
        output['feat'] = torch.stack(feats, dim=0)
        output['label'] = int(label)
        output['filename'] = data[0]['filename']
        return output

    # def read_seq(self, wsi):
    #     outputs = []
    #     for i in range(self.seq_len):
    #         outputs.append(next(wsi))

    #     return outputs

    def read_feat(self, patch_id):
        with self.env.begin(write=False) as txn:
            img_stream = txn.get(patch_id.encode())
            feat = unpack('1024f', img_stream)
            feat = torch.tensor(feat)

        return feat



    def read_seq(self, wsi):

        feats = []
        label = 0
        filename = ''
        # with self.env.begin(write=False) as txn:
        for idx, d in enumerate(wsi):
            patch_id = d['patch_id']

            feat = self.cache.get(patch_id, None)

            if feat is None:
               feat = self.read_feat(patch_id)
               self.cache[patch_id] = feat

            # feat = self.read_feat(patch_id)
            # img_stream = txn.get(patch_id.encode())
            # # feat = self.data[patch_id]

            # # feat = unpack('384f', img_stream)
            # feat = unpack('1024f', img_stream)
            # feat = torch.tensor(feat)
            feats.append(feat)
            label = d['label']
            filename = d['filename']
            # print(label)

            # img = np.frombuffer(img_stream, np.uint8)
            # In the case of color images, the decoded images will have the channels stored in B G R order.
            # img = cv2.imdecode(img, -1)  # most time is consum
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if idx == self.seq_len - 1:
                break

        output = {}
        output['feat'] = torch.stack(feats, dim=0)
        # output['feat'] = torch.tensor(feats)
        output['label'] = int(label)
        output['filename'] = filename
        return output

    def shift_wsi(self, batch_wsi, num_patches):
        random.seed(self.seed)
        # shift 1 or self.seq_len - 1 times
        # shift = random.randint(1,  - 1)
        # print('totoal {} number of wsis , shift {} number of tokens seq_len {} '.format(len(batch_wsi), shift, self.seq_len))
        tmp = []
        for wsi, num_patch in zip(batch_wsi, num_patches):

            shift = random.randint(0, num_patch - 1)
            for i in range(shift):
                next(wsi)

            tmp.append(wsi)

        return tmp

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        # create a new list to avoid change the self.orig_wsis
        # during each epoch
        wsis = []
        for wsi in self.orig_wsis:
            wsis.append(wsi)

        wsis = self.shuffle(wsis)

        # wsis = self.split_wsis(wsis) # used for ddp training
        # wsis = self.orgnize_wsis(wsis)
        # global_seq_len = self.cal_seq_len(wsis)

        if self.data_set != 'train':
            wsis = self.set_direction(wsis, direction=0)
            if self.drop_last:
                raise ValueError('during inference, the drop_last should not be set to true')
        else:
            wsis = self.set_random_direction(wsis)

        count = 0
        for idx in range(0, len(wsis), self.batch_size):#0
            # use cache to store the current wsi feats
            # since we are using a cyclic training procedure
            # features will be read multiple times from disk
            # therefore, cache current wsis will save some
            # I/O time.
            # The whole dataset is quit large 30GB - 100GB
            # storing all the dataset in the RAM would require
            # much more memory
            self.cache = {}

            # add seeds here to avoid different seed value for
            # different workers if we set seed += 1024 at the
            # end of each data = next(x) loop (count might not
            # be divided by each )
            self.seed += 1024

            batch_wsi = wsis[idx : idx + self.batch_size]

            # assert len(batch_wsi) == self.batch_size
            if len(batch_wsi) < self.batch_size:
                if self.drop_last:
                    continue

            num_patches = [w.num_patches for w in batch_wsi]
            batch_wsi = [self.cycle(x) for x in batch_wsi]

            if self.data_set == 'train':
                batch_wsi = self.shift_wsi(batch_wsi, num_patches)


            # max_len_idx = idx // self.batch_size
            # mex_len = self.wsi_len

            # if not global_seq_len[max_len_idx]:
                # warnings.warn('max batch len equals 0')
                # continue

            # max_len = global_seq_len[max_len_idx]

            # if wsi len is not divisible by num_workers,
            # the last few elements will
            # change the order of reading next round
            # set a global counter to eliminate this issue
            for patch_idx in range(0, self.wsi_len, self.seq_len):#104

                    outputs = []
                    for wsi in batch_wsi:

                        # data = next(wsi)
                        data = self.read_seq(wsi)
                        # print(len(list(self.cache.keys())))

                        if worker_info is not None:
                            if count % worker_info.num_workers != worker_info.id:
                                continue

                            #data['worker_id'] = worker_info.id
                            #data['count'] = self.count
                            #data['patch_idx'] = patch_idx
                            #data['seed'] = tmp_seed
                            # data['dir'] = tmp_dircts
                        # data = self.read_img(data)

                        # if patch_idx < max_len - 1:
                        # if patch_idx < self.wsi_len - 1:
                        if patch_idx < self.wsi_len - self.seq_len:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        outputs.append(data)

                    count += 1

                    if outputs:
                        data = default_collate(outputs)
                        # print(data['label'])
                        yield data['feat'], data['label'], data['filename'], data['is_last']

# dataset = CAM16(
#     data_set='test',
#     fold=0,
#     batch_size=1,
#     drop_last=False,
#     allow_reapt=False,
#     dist=None,
#     # direction=1
# )

# print(dataset)

# for idx, d in enumerate(dataset):
#     # output(d)
#     d1, d2, d3, d4 = d
#     pass
#     # print(d['feat'].shape)
#     # print('is_last', d['is_last'])
#     # print(d['label'])
#     # print(idx)



#class CAM16(WSIDataset):
#    # def __init__(self, data_set, lmdb_path, batch_size, drop_last=False, allow_reapt=False, transforms=None, dist=None):
#    # def __init__(self, data_set, fold, batch_size, drop_last=False, allow_reapt=False, dist=None):
#    def __init__(self, data, data_set, fold, batch_size, drop_last=False, allow_reapt=False, dist=None):
#        """the num_worker of each CAMLON16 dataset is one, """
#        # assert data_set in ['train', 'val']
#        print(self)
#
#
#        self.data = data
#
#        self.batch_size = batch_size
#        self.drop_last = drop_last
#        self.allow_reapt = allow_reapt
#        self.data_set = data_set
#        self.fold = fold
#        # self.trans = transforms
#        from conf.camlon16 import settings
#        self.settings = settings
#
#        # self.orig_wsis = self.get_wsis(data_set=data_set)
#        self.orig_wsis = self.get_wsis(data_set=self.data_set, fold=self.fold)
#        print('total number of wsis: {}'.format(len(self.orig_wsis)))
#        # self.env = lmdb.open(settings.feat_dir, readonly=True, lock=False)
#        self.seed = 52
#
#        self.dist = dist
#
#        self.max_len = 43950
#        self.seq_len = 1024
#        num_chunks = math.ceil(self.max_len / self.seq_len)
#        # self.wsi_len = 79872
#        self.wsi_len = num_chunks * 2 * 1024
#
#
#    def split_wsis(self, wsis):
#        if not self.dist.is_initialized():
#            return wsis
#        else:
#            """get wsis for each gpu"""
#            rank = self.dist.get_rank()
#            num_replicas = self.dist.get_world_size()
#            num_samples = math.ceil(len(wsis) / num_replicas)
#            subsample = wsis[rank * num_samples: (rank + 1) * num_samples]
#
#            # make sure each gpu acclocated the same number of wsis
#            if len(subsample) < num_samples:
#                diff = num_samples - len(subsample)
#                subsample.extend(wsis[:diff])
#
#            return subsample
#
#
#    def get_wsis(self, data_set, fold):
#        # wsis = self.data_set()
#        # from conf.camlon16 import settings
#        file_list = pd.read_csv(self.settings.file_list_csv)
#        file_list['slide_id'] = file_list['slide_id'].apply(
#            lambda x: os.path.splitext(x)[0]
#        )
#        # print(file_list['slide_id'])
#        # splits = os.path.join(os.pat)
#        # split_file = os.path.join('datasets', 'splits', 'cam16', 'splits_{}.csv'.format(fold))
#        # pd.set_option('display.max_rows', None)
#        split_file = os.path.join(self.settings.split_dir, 'splits_{}.csv'.format(fold))
#        splits = pd.read_csv(split_file)
#        # data_list = splits[data_set]
#        # if data_set != 'test':
#        train_split = splits['train'].dropna()
#        val_split = splits['val'].dropna()
#        test_split = splits['test'].dropna()
#        # print(data_list)
#        # print(val_split, train_split, test_split)
#
#
#        if data_set != 'test':
#            mask1 = file_list['slide_id'].isin(train_split)
#            mask2 = file_list['slide_id'].isin(val_split)
#            slide_ids = file_list['slide_id'][mask1 | mask2]
#        else:
#            mask1 = file_list['slide_id'].isin(test_split)
#            slide_ids = file_list['slide_id'][mask1]
#
#
#        # print(slide_ids)
#        wsis = []
#        for slide in slide_ids:
#            # print(i, '333')
#            json_path = os.path.join(self.settings.json_dir, slide + '.json')
#            print(json_path)
#            wsis.append(
#                WSIJSON(
#                    json_path=json_path,
#                    direction=0,
#                    )
#            )
#
#
#        # mask = file_list.isin(sli)
#        # print(len(slide_ids))
#
#        # data = file_list[mask]
#        # print(data)
#        # return data
#        return wsis
#
#
#
#
#    def set_direction(self, wsis, direction):
#        for wsi in wsis:
#            wsi.direction = direction
#
#        return wsis
#
#    def set_random_direction(self, wsis):
#        random.seed(self.seed)
#        for wsi in wsis:
#            direction = random.randint(0, 7)
#            wsi.direction = direction
#        return wsis
#
#    def orgnize_wsis(self, orig_wsis):
#        wsis = []
#        # when batch size is larger than the total num of wsis
#        if self.batch_size > len(orig_wsis):
#            if self.allow_reapt:
#                wsis = orig_wsis
#                while len(wsis) != self.batch_size:
#                    wsis.extend(random.sample(orig_wsis, k=1))
#            else:
#                raise ValueError('allow_reapt should be True when batch_size is larger than the whole wsis')
#
#        else:
#            remainder = len(orig_wsis) % self.batch_size
#            # if the total number of wsis is not divisible by batch_size
#            if remainder > 0:
#                # if we do not drop the last, we randomly select "self.batch_size - remainder" number of
#                # samples add to the orig_
#                random.seed(self.seed)
#                if not self.drop_last:
#                    wsis = orig_wsis
#                    # for wsi in random.sample(self.orig_wsis, k=self.batch_size - remainder):
#                    for wsi in random.sample(orig_wsis, k=self.batch_size - remainder):
#                        wsis.append(wsi)
#
#                else:
#                    # if drop last, we randomly sample "total number of self.orig_wsis - remainer" number
#                    # of wsis
#                    wsis = random.sample(orig_wsis, k=len(orig_wsis) - remainder)
#                    assert len(wsis) == len(orig_wsis) - remainder
#            else:
#                wsis = orig_wsis
#        return wsis
#
#
#    def shuffle(self, wsis):
#        """manually shuffle all the wsis, because when num_workers > 0,
#        the copy of dataset wont update in the main process """
#        random.seed(self.seed)
#        random.shuffle(wsis)
#
#        return wsis
#
#    def cal_seq_len(self, wsis):
#        outputs = []
#
#        for idx in range(0, len(wsis), self.batch_size):
#
#            batch_wsi = wsis[idx : idx + self.batch_size]
#            max_len = max([wsi.num_patches for wsi in batch_wsi])
#
#            outputs.append(max_len)
#
#        return outputs
#
#    def cycle(self, iterable):
#        while True:
#            for data in iterable:
#                yield data
#
#    def read_img(self, data):
#
#        output = {}
#        # with self.env.begin(write=False) as txn:
#        feats = []
#        label = 0
#        for d in data:
#            patch_id = d['patch_id']
#            # img_stream = txn.get(patch_id.encode())
#            feat = self.data[patch_id]
#
#            # feat = unpack('384f', img_stream)
#            # feat = unpack('1024f', img_stream)
#            feats.append(feat)
#            label = d['label']
#                # print(label)
#
#                # img = np.frombuffer(img_stream, np.uint8)
#                # In the case of color images, the decoded images will have the channels stored in B G R order.
#                # img = cv2.imdecode(img, -1)  # most time is consum
#                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#
#        # output['feat'] = torch.tensor(feats)
#        # output['feat'] = torch.tensor(feats)
#        output['feat'] = torch.stack(feats, dim=0)
#        output['label'] = int(label)
#        output['filename'] = data[0]['filename']
#        return output
#
#    def read_seq(self, wsi):
#        outputs = []
#        for i in range(self.seq_len):
#            outputs.append(next(wsi))
#
#        return outputs
#
#    def shift_wsi(self, batch_wsi, num_patches):
#        random.seed(self.seed)
#        # shift 1 or self.seq_len - 1 times
#        # shift = random.randint(1,  - 1)
#        # print('totoal {} number of wsis , shift {} number of tokens seq_len {} '.format(len(batch_wsi), shift, self.seq_len))
#        tmp = []
#        for wsi, num_patch in zip(batch_wsi, num_patches):
#
#            shift = random.randint(0, num_patch - 1)
#            for i in range(shift):
#                next(wsi)
#
#            tmp.append(wsi)
#
#        return tmp
#
#    def __iter__(self):
#
#        worker_info = torch.utils.data.get_worker_info()
#
#        # create a new list to avoid change the self.orig_wsis
#        # during each epoch
#        wsis = []
#        for wsi in self.orig_wsis:
#            wsis.append(wsi)
#
#        wsis = self.shuffle(wsis)
#        # wsis = self.split_wsis(wsis) # used for ddp training
#        # wsis = self.orgnize_wsis(wsis)
#        # global_seq_len = self.cal_seq_len(wsis)
#
#        if self.data_set != 'train':
#            wsis = self.set_direction(wsis, direction=0)
#            if self.drop_last:
#                raise ValueError('during inference, the drop_last should not be set to true')
#        else:
#            wsis = self.set_random_direction(wsis)
#
#        count = 0
#        for idx in range(0, len(wsis), self.batch_size):#0
#
#            # add seeds here to avoid different seed value for
#            # different workers if we set seed += 1024 at the
#            # end of each data = next(x) loop (count might not
#            # be divided by each )
#            self.seed += 1024
#
#            batch_wsi = wsis[idx : idx + self.batch_size]
#
#            # assert len(batch_wsi) == self.batch_size
#            if len(batch_wsi) < self.batch_size:
#                if self.drop_last:
#                    continue
#
#            num_patches = [w.num_patches for w in batch_wsi]
#            batch_wsi = [self.cycle(x) for x in batch_wsi]
#
#            if self.data_set == 'train':
#                batch_wsi = self.shift_wsi(batch_wsi, num_patches)
#
#
#
#            # max_len_idx = idx // self.batch_size
#            # mex_len = self.wsi_len
#
#            # if not global_seq_len[max_len_idx]:
#                # warnings.warn('max batch len equals 0')
#                # continue
#
#            # max_len = global_seq_len[max_len_idx]
#
#            # if wsi len is not divisible by num_workers,
#            # the last few elements will
#            # change the order of reading next round
#            # set a global counter to eliminate this issue
#            for patch_idx in range(0, self.wsi_len, self.seq_len):#104
#
#                    outputs = []
#                    for wsi in batch_wsi:
#
#                        # data = next(wsi)
#                        data = self.read_seq(wsi)
#
#                        if worker_info is not None:
#                            if count % worker_info.num_workers != worker_info.id:
#                                continue
#
#                            #data['worker_id'] = worker_info.id
#                            #data['count'] = self.count
#                            #data['patch_idx'] = patch_idx
#                            #data['seed'] = tmp_seed
#                            # data['dir'] = tmp_dircts
#                        data = self.read_img(data)
#
#                        # if patch_idx < max_len - 1:
#                        # if patch_idx < self.wsi_len - 1:
#                        if patch_idx < self.wsi_len - self.seq_len:
#                            data['is_last'] = 0
#                        else:
#                            data['is_last'] = 1
#
#                        outputs.append(data)
#
#                    count += 1
#
#                    if outputs:
#                        data = default_collate(outputs)
#                        # print(data['label'])
#                        yield data['feat'], data['label'], data['filename'], data['is_last']
#
