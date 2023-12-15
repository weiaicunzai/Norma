

# from torch.utils.data import Data
# import glob
import os
# from typing import Any

# import torch
# from torch.utils.data import Dataset, IterableDataset
# import openslide
# import cv2
# import queue
# import random
import csv
import torch
# import warnings

from torchvision import transforms


# from preprocess import utils
# from .wsi import WSI
# from itertools import cycle
#from .dataloader import WSIDataLoader
#from .dist_dataloader import DistWSIDataLoader

# from dataset.aa import WSIDataLoader
# from .dataloader import WSIDataLoader
from dataset.dataloader import WSIDataLoader
# from dataset.aa import DistWSIDataLoader
from .dist_dataloader import DistWSIDataLoader

class CAMLON16Label:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        # for line in open()

        self.name2label = {}

        with open(csv_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            # print(list(spamreader))
            for row in spamreader:
                row = row[0].split(',')
                # print(row)
                if row[1] == 'Normal':
                    label = 0
                elif row[1] == 'Tumor':
                    label = 1
                # else:
                self.name2label[row[0]] = label

            # for row in spamreader:
                # print(row)

        # print(self.name2label)
    def __call__(self, filename):
        basename = os.path.basename(filename)
        # print(basename)
        if 'tumor' in basename:
            return 1
        elif 'normal' in basename:
            return 0
        elif 'test' in basename:
            return self.name2label[os.path.basename(filename).split('.')[0]]
        else:
            raise ValueError('wrong filename {}'.format(filename))





#class CAMLON16(IterableDataset):
#    # def __init__(self, wsi_img_dir, wsi_mask_dir, patch_size=256, at_mag=20, random_rotate=False):
#    def __init__(self, wsis, batch_size, drop_last):
#        """the num_worker of each CAMLON16 dataset is one, """
#        # self.wis_img_dir =  wsi_img_dir
#        # self.wsi_mask_dir = wsi_mask_dir
#        self.wsis = wsis
#        # self.patch_size = patch_size
#
#        # self.wsi_filenams = []
#        # self.wsis = []
#        self.batch_size = batch_size
#        self.drop_last = drop_last
#
#
#        self.global_seq_len = -1
#
#
#
#        # self.patch_size
#
#        # print(self.wis_img_path)
#        # for wsi_path in glob.iglob(os.path.join(self.wis_img_dir, '**', '*.tif'), recursive=True):
#
#        #     # self.wsi_filenames.append(i)
#        #     self.wsis.append(WSI(
#        #         wsi_path,
#        #         self.mask_path(wsi_path),
#        #         patch_size=self.patch_size,
#        #         at_mag=at_mag,
#        #         random_rotate=random_rotate
#        #     ))
#            # print(i)
#            # print(self.mask_path(i))
#
#        # self.start_idx = 0
#        # self.end_idx = len(self.wsi) - 1
#
#
#        # self.end_of_wsi = True # indicate whether the end of a wsi
#        # self.patch_scale = 0
#
#        # self.wsi = None
#        # self.mask = None
#        # self.scale = None
#        # self.coordinates = None
#
#        # self.row_idx = 0
#        # self.col_idx = 0
#
#
#    # def mask_path(self, wsi_path):
#    #     mask_path = wsi_path.replace('training', 'training_mask')
#    #     mask_path = mask_path.replace('.tif', '.png')
#    #     return mask_path
#
#    # def level_0_magnification(self, wsi):
#    #     if 'aperio.AppMag' in wsi.properties.keys():
#    #         level_0_magnification = int(wsi.properties['aperio.AppMag'])
#    #     elif 'openslide.mpp-x' in wsi.properties.keys():
#    #         level_0_magnification = 40 if int(float(wsi.properties['openslide.mpp-x']) * 10) == 2 else 20
#    #     else:
#    #         level_0_magnification = 40
#
#    #     return level_0_magnification
#
#
##    def get_patch_scale(self, wsi, mask):
##         # if is the start of a new wsi
##         # if self.end_of_wsi:
##             # start compute patch_scale
#
##             # get the wsi magnifiaction factor of dimension
##             level_0_mag = self.level_0_magnification(wsi)
#
##             # sample at 20x level
##             if level_0_mag == 40:
##                 level = 1
##             else:
##                 level = 0
#
##             # print(level)
##             (width, height) =  wsi.level_dimensions[level]
##             mag_20_resolution = (height, width)
##             # print(mag_20_resolution)
##             print(mag_20_resolution, mask.shape)
#
##             # use multiply to avoid float number comparison
##             assert mag_20_resolution[0] * mask.shape[1] == mag_20_resolution[1] * mask.shape[0]
#
##             scale = mag_20_resolution[0] / mask.shape[0]
#
##             return scale
#
#
#
#    # def __len__(self):
#    #     return len(self.wsi_fns)
#
#
#    # def get_coordinates(self, row, col, stride):
#    #     coords = queue.Queue()
#
#    #     for r_idx in range(0, row, stride):
#    #         for c_idx in range(0, col, stride):
#    #             coords.put((r_idx, c_idx))
#
#    #     return coords
#
#    # def init_wsi_variables(self, wsi, mask):
#
#    #     self.scale = self.get_patch_scale(self.wsi, self.mask)
#    #     # scaled_patch_size = self.patch_size / self.scale
#
#    #     row, col = self.mask.shape[:2]
#
#    #     # self.coords = self.get_coordinates()
#
#    #     self.scale = self.get_patch_scale(self.wsi, self.mask)
#    #     stride = int(self.patch_size / self.scale)
#    #     # print(stride)
#    #     self.coords = self.get_coordinates(row, col, stride)
#
#    # @property
#    # def patch_len_seq(self):
#    #     num_patch_seq = []
#
#    #     for idx in range(0, len(self.wsis), self.batch_size):
#
#    #         batch_wsi = self.wsis[idx : idx + self.batch_size]
#
#    #         if self.drop_last:
#    #             if len(batch_wsi) != self.batch_size:
#    #                 return
#    #         else:
#    #             for i in range(len(batch_wsi), self.batch_size):
#    #                 batch_wsi.append(self.wsis[i])
#
#    #         num_patch_seq.append(
#    #                 max([wsi.num_patches for wsi in batch_wsi])
#    #             )
#
#    #     return num_patch_seq
#
#    # def __getitem__(self, idx):
#    def __iter__(self):
#
#        # print('....................................')
#
#        for idx in range(0, len(self.wsis), self.batch_size):
#
#            batch_wsi = self.wsis[idx : idx + self.batch_size]
#
#
#            if self.drop_last:
#                if len(batch_wsi) != self.batch_size:
#                    return
#            else:
#                for i in range(len(batch_wsi), self.batch_size):
#                # for i in range(len(batch_wsi), self.start_idx - self.end_idx):
#                    batch_wsi.append(self.wsis[i])
#
#
#            assert len(batch_wsi) == self.batch_size
#
#            # max_batch_lenth = max([len(x) for x in batch_wsi])
#
#            print([x.wsi_name for x in batch_wsi])
#            batch_wsi = [cycle(x) for x in batch_wsi]
#            # for i in range(self.max_batch_len):
#
#            max_len_idx = idx // self.batch_size
#            if not self.global_seq_len[max_len_idx]:
#                warnings.warn('max batch len equals 0')
#                continue
#            for _ in  range(self.global_seq_len[idx // self.batch_size]):
#                # print('ccccccccccc', i)
#                # sleep_time = 0.005 * len(batch_wsi)
#                # time.sleep(sleep_time)
#
#                try:
#
#                    # print(batch_wsi)
#                    print(self.global_seq_len[idx // self.batch_size], idx // self.batch_size)
#                    yield [next(x) for x in batch_wsi]
#                except Exception as e:
#                    print('sss', e)
#
#
#
#
#
#
#
#        # random.shuffle(self.wsis)
#
#        # if self.end_of_wsi:
#        #     print('?????')
#        #     self.wsi = openslide.OpenSlide(self.wsi_fns[idx])
#        #     print(self.wsi_fns[idx])
#        #     self.mask = cv2.imread(self.mask_path(self.wsi_fns[idx]))
#        #     print(self.mask_path(self.wsi_fns[idx]))
#        #     # print(self.mask.shape, self.wsi)
#        #     # self.scale = self.get_patch_scale(self.wsi, self.mask)
#        #     # stride = self.patch_size / self.scale
#        #     # self.coords = self.get_coordinates(row, col, stride)
#        #     self.init_wsi_variables(self.wsi, self.mask)
#
#
#        # coord = self.coords.get()
#        # print(coord)
#        # scaled_patch_size = self.patch_size / self.scale
#        # print(scaled_patch_size)
#
#        # row, col = self.mask.shape[:2]
#        # if self.col_idx * scaled_patch_size > col - 1:
#        #     self.col_idx = 0
#        #     self.row_idx += 1
#
#        # if self.row_idx * scaled_patch_size > row - 1:
#        #     self.end_of_wsi = True
#
#        # self.mask[ro]
#            # print(self.scale, self.scale)
#
#
#        # get_patch_scale()
#
#
#
## path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training'
## mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/'
## dataset = CAMLON16(wsi_img_path=path, wsi_mask_path=mask_path)
#
#
## mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_153.png'
## wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_153.tif'
#
## mask = cv2.imread(mask_path, -1)
#
## # wsi = openslide.OpenSlide()
#
## stride = 8
#
## dataset = CAMLON16(wsi_path, mask_path)
## print(dataset)
#
## mask
#
## import time
## for row in range(0, mask.shape[0], 8):
##     for col in range(0, mask.shape[1], 8):
##         print(row,col)
#

def get_num_classes(dataset_name):
    if dataset_name == 'cam16':
        return 2

def A_trans(img_set):

    import albumentations as A
    import albumentations.pytorch as AP
    img_size = (256, 256)
    if img_set == 'train':
        trans = A.Compose([
                A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.5, 2.0), ratio=(1, 1), always_apply=True),
                # transforms.RandomChoice
                A.OneOf(
                    [
                        # nothing:
                        A.Compose([]),

                        # h:
                        # transforms.RandomHorizontalFlip(p=1),
                        A.HorizontalFlip(p=1),

                        # v:
                        # transforms.RandomVerticalFlip(p=1),
                        A.VerticalFlip(p=1),

                        # hv:
                        # transforms.Compose([
                        A.Compose([
                               #transforms.RandomVerticalFlip(p=1),
                               #transforms.RandomHorizontalFlip(p=1),
                               A.VerticalFlip(p=1),
                               A.HorizontalFlip(p=1),
                        ]),

                         #r90:
                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                        # transforms.MyRotate90(degrees=(90, 90), expand=True, p=1),
                        # transforms.MyRotate90(p=1),
                        # transforms.RandomRotation(degrees=(90, 90)),
                        A.Rotate([90,90]),

                        # #r90h:
                        # transforms.Compose([
                        A.Compose([
                            # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            # transforms.MyRotate90(p=1),
                            A.Rotate([90, 90]),
                            # transforms.RandomHorizontalFlip(p=1),
                            A.HorizontalFlip(p=1),
                        ]),

                        # #r90v:
                        # transforms.Compose([
                        A.Compose([
                            # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            # transforms.MyRotate90(p=1),
                            # transforms.RandomRotation(degrees=(90, 90)),
                            A.Rotate([90, 90]),
                            # transforms.RandomVerticalFlip(p=1),
                            A.VerticalFlip(p=1)
                        ]),

                        # #r90hv:
                        # transforms.Compose([
                        A.Compose([
                            # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            # transforms.MyRotate90(p=1),
                            # transforms.RandomRotation(degrees=(90, 90)),
                            A.Rotate([90, 90]),
                            #transforms.RandomHorizontalFlip(p=1),
                            #transforms.RandomVerticalFlip(p=1),
                            A.HorizontalFlip(p=1),
                            A.VerticalFlip(p=1),
                        ]),
                    ]
                ),
                A.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4, hue=0.1, p=0.5),
                A.Compose([
                    A.ToGray(p=1),
                    # A.ToRGB(p=1)
                ], p=0.1),
                A.Normalize(mean=(0.62438617, 0.45624277, 0.64247613), std=(0.25213961, 0.27547218, 0.21659795)),
                AP.transforms.ToTensorV2()
        ])

    else:
        trans = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                # A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.5, 2.0), ratio=(1, 1), always_apply=True),
                # transforms.RandomChoice
               # A.OneOf(
               #     [
               #         # nothing:
               #         A.Compose([]),

               #         # h:
               #         # transforms.RandomHorizontalFlip(p=1),
               #         A.HorizontalFlip(p=1),

               #         # v:
               #         # transforms.RandomVerticalFlip(p=1),
               #         A.VerticalFlip(p=1),

               #         # hv:
               #         # transforms.Compose([
               #         A.Compose([
               #                #transforms.RandomVerticalFlip(p=1),
               #                #transforms.RandomHorizontalFlip(p=1),
               #                A.VerticalFlip(p=1),
               #                A.HorizontalFlip(p=1),
               #         ]),

               #          #r90:
               #         # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
               #         # transforms.MyRotate90(degrees=(90, 90), expand=True, p=1),
               #         # transforms.MyRotate90(p=1),
               #         # transforms.RandomRotation(degrees=(90, 90)),
               #         A.Rotate([90,90]),

               #         # #r90h:
               #         # transforms.Compose([
               #         A.Compose([
               #             # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
               #             # transforms.MyRotate90(p=1),
               #             A.Rotate([90, 90]),
               #             # transforms.RandomHorizontalFlip(p=1),
               #             A.HorizontalFlip(p=1),
               #         ]),

               #         # #r90v:
               #         # transforms.Compose([
               #         A.Compose([
               #             # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
               #             # transforms.MyRotate90(p=1),
               #             # transforms.RandomRotation(degrees=(90, 90)),
               #             A.Rotate([90, 90]),
               #             # transforms.RandomVerticalFlip(p=1),
               #             A.VerticalFlip(p=1)
               #         ]),

               #         # #r90hv:
               #         # transforms.Compose([
               #         A.Compose([
               #             # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
               #             # transforms.MyRotate90(p=1),
               #             # transforms.RandomRotation(degrees=(90, 90)),
               #             A.Rotate([90, 90]),
               #             #transforms.RandomHorizontalFlip(p=1),
               #             #transforms.RandomVerticalFlip(p=1),
               #             A.HorizontalFlip(p=1),
               #             A.VerticalFlip(p=1),
               #         ]),
               #     ]
               # ),
               # A.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4, hue=0.1, p=0.5),
               # A.Compose([
               #     A.ToGray(p=1),
               #     # A.ToRGB(p=1)
               # ], p=0.1),
                A.Normalize(mean=(0.62438617, 0.45624277, 0.64247613), std=(0.25213961, 0.27547218, 0.21659795)),
                AP.transforms.ToTensorV2()
        ])
    return trans

def build_transforms(img_set):
    assert img_set in ['val', 'train', 'test']

    img_size=(256, 256)
    if img_set == 'train':

        #trans = transforms.Compose([
        #        # transforms.RandomRotation(30),
        #        transforms.RandomResizedCrop(img_size, scale=(0.5, 2.0), ratio=(1,1)),
        #        # transforms.RandomApply(
        #        #     transforms=[transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
        #        #     p=0.5
        #        # ),


        #        transforms.RandomChoice
        #            (
        #                [
        #                    # nothing:
        #                    transforms.Compose([]),

        #                    # h:
        #                    transforms.RandomHorizontalFlip(p=1),

        #                    # v:
        #                    transforms.RandomVerticalFlip(p=1),

        #                    # hv:
        #                    transforms.Compose([
        #                           transforms.RandomVerticalFlip(p=1),
        #                           transforms.RandomHorizontalFlip(p=1),
        #                    ]),

        #                     #r90:
        #                    # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
        #                    # transforms.MyRotate90(degrees=(90, 90), expand=True, p=1),
        #                    # transforms.MyRotate90(p=1),
        #                    transforms.RandomRotation(degrees=(90, 90)),

        #                    # #r90h:
        #                    transforms.Compose([
        #                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
        #                        # transforms.MyRotate90(p=1),
        #                        transforms.RandomRotation(degrees=(90, 90)),
        #                        transforms.RandomHorizontalFlip(p=1),
        #                    ]),

        #                    # #r90v:
        #                    transforms.Compose([
        #                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
        #                        # transforms.MyRotate90(p=1),
        #                        transforms.RandomRotation(degrees=(90, 90)),
        #                        transforms.RandomVerticalFlip(p=1),
        #                    ]),

        #                    # #r90hv:
        #                    transforms.Compose([
        #                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
        #                        # transforms.MyRotate90(p=1),
        #                        transforms.RandomRotation(degrees=(90, 90)),
        #                        transforms.RandomHorizontalFlip(p=1),
        #                        transforms.RandomVerticalFlip(p=1),
        #                    ]),
        #                ]
        #            ),

        #        transforms.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=10),
        #        # transforms.RandomCrop(img_size),
        #        #transforms.RandomApply(
        #        #    transforms=[transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
        #        #    p=0.3
        #        #),
        #        #transforms.RandomGrayscale(p=0.1),
        #        # transforms.RandomHorizontalFlip(p=0.5),
        #        # transforms.RandomVerticalFlip(p=0.5),
        #        transforms.ToTensor(),
        #        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #])


        trans = A_trans(img_set=img_set)

        # trans = transforms.Compose([
        #     transforms.RandomRotation(30),
        #     transforms.RandomResizedCrop(img_size, scale=(0.5, 2.0), ratio=(1,1)),
        #     # transforms.RandomCrop(img_size),
        #     transforms.RandomApply(
        #         transforms=[transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
        #         p=0.3
        #     ),
        #     transforms.RandomGrayscale(p=0.1),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])


    elif img_set in ['val', 'test']:
        #trans = transforms.Compose([
        #    # transforms.RandomRotation(30),
        #    # transforms.RandomResizedCrop(img_size, scale=(0.5, 2.0), ratio=(1,1)),
        #    # # transforms.RandomCrop(img_size),
        #    # transforms.RandomApply(
        #    #     transforms=[transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
        #    #     p=0.3
        #    # ),
        #    # transforms.RandomGrayscale(p=0.1),
        #    # transforms.RandomHorizontalFlip(p=0.5),
        #    # transforms.RandomVerticalFlip(p=0.5),
        #    transforms.Resize(img_size),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #])
        trans = A_trans(img_set)

    else:
        raise ValueError('wrong img_set value {}'.format(img_set))


    return trans







def build_dataloader(dataset_name, img_set, dist, batch_size, num_workers, num_gpus=None, all=True, drop_last=True):
    assert isinstance(dist, bool)
    assert img_set in ['val', 'train', 'test']

    trans = build_transforms(img_set)



    # if dataset_name == 'cam16_seq':
    if dataset_name == 'cam16':
        from dataset.wsi_reader import camlon16_wsis
        from dataset.wsi_dataset import WSIDataset
        # from dataset.wsi_dataset import WSIDatasetLMDB
        from conf.camlon16 import settings

        if img_set == 'train':
            direction = -1
        else:
            direction = 0
        wsis = camlon16_wsis(img_set, direction=direction)

        print(all)
        if not all:
            tmp = []
            for wsi in wsis:
                wsi.patch_level()
                tmp.append(wsi)
            wsis = tmp

        print('xxxxxx', )
        dataset_cls = WSIDataset

    # if dataset_name == 'cam16_map':
    #     from dataset.vit_lmdb import CAM16
    #     dataset = CAM16(trans=trans, image_set=img_set)
        # dataloader = torch.utils.data.DataLoader(
        #         dataset,
        #         batch_size=batch_size,
        #         num_workers=num_workers,
        #         shuffle=True
        #     )

        # if num_gpus > 1:
        #     torch
    if img_set == 'train':
        shuffle = True
        lmdb_path = settings.train_dirs['lmdb'][0]
        allow_repeat = True
    else:
        shuffle = False
        lmdb_path = settings.test_dirs['lmdb'][0]
        allow_repeat = False

    if dist:
        dataloader = DistWSIDataLoader(
            lmdb_path=lmdb_path,
            wsis=wsis,
            # batch_size=17,
            batch_size=batch_size,
            num_gpus=num_gpus,
            cls_type=dataset_cls,
            num_workers=num_workers,
            transforms=trans,
            shuffle=shuffle,
        )

    else:
        # print('?????')
        print(WSIDataLoader)
        dataloader = WSIDataLoader(
            wsis,
            shuffle=shuffle,
            batch_size=batch_size,
            cls_type=WSIDataset,
            pin_memory=True,
            num_workers=num_workers,
            transforms=trans,
            allow_repeat=allow_repeat,
            # drop_last=True,
            drop_last=drop_last,
        )


    print(dataloader)
    return dataloader