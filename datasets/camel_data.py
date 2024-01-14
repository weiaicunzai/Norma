import random
import torch
import pandas as pd
from pathlib import Path
import pickle

import numpy as np
import torch.utils.data as data
from torch.utils.data import dataloader


class CamelData1(data.Dataset):
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



class CamelData(data.IterableDataset):
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
#
#
#        # return features, label, slide_id
#
#
#    #def __getitem__(self, idx):
#    #    slide_id = self.data[idx]
#    #    label = int(self.label[idx])
#    #    # full_path = Path(self.feature_dir) / f'{slide_id}.pt'
#    #    # features = torch.load(full_path)
#    #    # features = self.feats[slide_id]
#    #    features = self.read_data(slide_id)
#
#    #    #----> shuffle
#    #    if self.shuffle == True:
#    #        index = [x for x in range(features.shape[0])]
#    #        random.shuffle(index)
#    #        features = features[index]
#
#    #    # features = features[:4]
#
#    #    # print(features[:, 4], self.wsi_length)
#    #    # features = self.pad_seq(features)
#    #    # print(features[:, 4], 'after')
#
#
#    #    return features, label, slide_id
#