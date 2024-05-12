import os
import glob
import csv


# from dataset.utils import BRACLabel

# import sys; sys.exit()

_dataset_path = '/data/smb/syh/WSI_cls/TCGA_BRCA/'
# _dataset_path = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/'

# train_dirs = {
#     # 'wsis': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/', '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/tumor/'],
#     # 'jsons': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_json/normal/patch_size_512_at_mag_5', '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_json/tumor/patch_size_512_at_mag_5/']
#     'wsis': [os.path.join(_dataset_path, 'training/normal/'), os.path.join(_dataset_path, 'training/tumor/')],
#     # 'jsons': [os.path.join(_dataset_path, 'training_json/normal/patch_size_512_at_mag_5'), os.path.join(_dataset_path, 'training_json/tumor/patch_size_512_at_mag_5/')]
#     'jsons': [os.path.join(_dataset_path, 'training_json/normal/patch_size_512_at_mag_20'), os.path.join(_dataset_path, 'training_json/tumor/patch_size_512_at_mag_20/')]
# }

# test_dirs = {
#     # 'wsis': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/images/'],
#     'wsis': [os.path.join(_dataset_path, 'testing/images/')],
#     # 'jsons' : [os.path.join(_dataset_path, 'testing/jsons/patch_size_512_at_mag_5/')]
#     'jsons' : [os.path.join(_dataset_path, 'testing/jsons/patch_size_512_at_mag_20/')]
# }

train_dirs = {
    # 'wsis': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/', '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/tumor/'],
    # 'jsons': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_json/normal/patch_size_512_at_mag_5', '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_json/tumor/patch_size_512_at_mag_5/']
    'wsis': [os.path.join(_dataset_path, 'img')],
    # 'jsons': [os.path.join(_dataset_path, 'training_json/normal/patch_size_512_at_mag_5'), os.path.join(_dataset_path, 'training_json/tumor/patch_size_512_at_mag_5/')]
    'jsons': [os.path.join(_dataset_path, 'json/patch_size_256_at_mag_20')],
    # 'lmdb' : [os.path.join(_dataset_path, 'training_lmdb')],
    'lmdb' : [os.path.join(_dataset_path, 'training_feat1')],
    # 'lmdb' : ['/dev/shm/by/training_lmdb/'],
    'patch_level': [os.path.join(_dataset_path, 'training_json/tumor/patch_size_512_at_mag_20_patch_label')],
    'anno': [os.path.join(_dataset_path, 'training/lesion_anno')]
}

# test_dirs = {
#     # 'wsis': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/images/'],
#     'wsis': [os.path.join(_dataset_path, 'testing/images/')],
#     # 'jsons' : [os.path.join(_dataset_path, 'testing/jsons/patch_size_512_at_mag_5/')]
#     'jsons' : [os.path.join(_dataset_path, 'testing/jsons/patch_size_512_at_mag_20/')]
# }

test_dirs = {
    # 'wsis': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/images/'],
    'wsis': [os.path.join(_dataset_path, 'img')],
    # 'jsons' : [os.path.join(_dataset_path, 'testing/jsons/patch_size_512_at_mag_5/')]
    # 'jsons' : [os.path.join(_dataset_path, 'testing/jsons/patch_size_512_at_mag_20/')],
    'jsons': [os.path.join(_dataset_path, 'json/patch_size_256_at_mag_20')],
    # 'lmdb' : [os.path.join(_dataset_path, 'testing_lmdb/')],
    'lmdb' : [os.path.join(_dataset_path, 'testing_feat1')],
    # 'lmdb' : ['/dev/shm/by/testing_lmdb/'],
    'patch_level': [os.path.join(_dataset_path, 'testing/jsons/patch_size_512_at_mag_20_patch_label')],
    'anno': [os.path.join(_dataset_path, 'testing/lesion')],
    'file': os.path.join(_dataset_path, 'dataset/dataset_csv/brac/brac.csv')
}





# _csv_file='/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/reference.csv'
# _csv_file= os.path.join(_dataset_path, 'dataset/dataset_csv/brac/brac.csv')
_csv_file= 'dataset/dataset_csv/brac/brac.csv'

# brac_label_fn = BRACLabel(csv_file=_csv_file)


# def camlon16_wsi_filenames(data_set):
#     outputs = []
#     if data_set == 'train':
#         dirs = _train_dirs
#     else:
#         dirs = _test_dirs


#     for wsi_dir, json_dir in zip(dirs['wsis'], dirs['jsons']):
#         for wsi_path in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=true):
#             basename = os.path.basename(wsi_path)
#             json_path = os.path.join(json_dir, basename.replace('.tif', '.json'))
#             outputs.append({
#                 'wsi': wsi_path,
#                 'json': json_path
#             })

#     return outputs


    # if  data_set == 'test':
    #     wsi_dirs = _train_dirs['wsis']
    #     for wsi_dir in _test_dirs['wsis']:
    #         # for wsi_dir in _test_dirs['wsis']:
    #         for wsi in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
    #             outputs.append(wsi)

    # return outputs



class Settings:
    def __init__(self):
        # self.label_fn = brac_label_fn
        # self.train_dirs = train_dirs
        # self.test_dirs = test_dirs
        self.mag = 20
        self.patch_size = 256
        self.root = '/data/smb/syh/WSI_cls/brac/'
        self.wsi_dir = os.path.join(self.root, 'img')
        self.mask_dir = os.path.join(self.root, 'mask')
        self.json_dir = os.path.join(self.root, 'json', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        self.file_list_csv = 'datasets/dataset_csv/brac/brac.csv'
        self.patch_dir = os.path.join(self.root, 'patch', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        # self.feat_dir = os.path.join(self.root, 'feat_dino', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        self.feat_dir = os.path.join(self.root, 'feat_ctranspath', 'patch_size_{}_at_mag_{}'.format(self.patch_size, self.mag))
        self.split_dir = 'datasets/splits/brac'
        self.num_classes = 2
        self.max_len = 67063
        self.ignore_label = -1



    # def file_list(self):
    #     # file = os.path.join(, 'dataset/dataset_csv/brac/brac.csv')
    #     with open(self.file_list_csv, 'r') as csv_file:
    #         for row in csv.DictReader(csv_file):
    #             row['slide_id'] = os.path.join(_dataset_path, row['slide_id'])
    #             yield row




settings = Settings()