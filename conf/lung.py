import os
import glob
import csv


class Settings:
    def __init__(self):
        self.mag = 20
        self.patch_size = 256
        self.root = '/data/smb/syh/WSI_cls/lung/'
        self.wsi_dir = os.path.join(self.root, 'img')
        self.mask_dir = os.path.join(self.root, 'mask')
        self.json_dir = os.path.join(self.root, 'json', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        self.file_list_csv = 'datasets/dataset_csv/lung/lung.csv'
        self.patch_dir = os.path.join(self.root, 'patch', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        # self.feat_dir = os.path.join(self.root, 'feat', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        # self.feat_dir = os.path.join(self.root, 'feat_ctranspath', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        # self.feat_dir = os.path.join(self.root, 'feat_lunti', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        self.feat_dir = os.path.join(self.root, 'uni', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        self.split_dir = 'datasets/splits/lung'
        self.num_classes = 2
        self.max_len = 46543
        self.ignore_label = -1


settings = Settings()