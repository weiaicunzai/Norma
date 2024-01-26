import os

class Settings:
    def __init__(self):

        self.mag = 20
        self.patch_size = 256
        self.root = '/data/smb/syh/WSI_cls/bracs/'
        self.wsi_dir = os.path.join(self.root, 'img')
        self.mask_dir = os.path.join(self.root, 'mask')
        self.json_dir = os.path.join(self.root, 'json', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        self.file_list_csv = 'datasets/dataset_csv/bracs/bracs.csv'
        self.patch_dir = os.path.join(self.root, 'patch', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        self.patch_label_dir = os.path.join(self.root, 'patch_label', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        self.feat_dir = os.path.join(self.root, 'feat', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        self.anno_dir = os.path.join(self.root, 'anno')
        self.split_dir = 'datasets/splits/cam16'
        self.num_classes = 2


settings = Settings()