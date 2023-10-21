
import os
import sys
import glob

sys.path.append(os.getcwd())
import cv2

# from datast.camlon16 import CAMLON16
from dataset import CAMLON16, WSI, CAMLON16Lable
# from .wsi import WSI


# mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/'
# wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/'

# mask = cv2.imread(mask_path, -1)

# # wsi = openslide.OpenSlide()

# stride = 8

# dataset = CAMLON16(wsi_path, mask_path)
# #print(dataset)
# #dataset[33]
# print(len(dataset))
# dataset[133]
def mask_path(wsi_path):
    mask_path = wsi_path.replace('training', 'training_mask')
    mask_path = mask_path.replace('.tif', '.png')
    return mask_path



def get_patch_sequqnce(dataset):

    res = []
    for wsi in dataset.wsis:
        res.append(wsi.num_patches)


    return res


def test_camlon16(wis_img_dir):

    label_fn = CAMLON16Lable(csv_file='/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/reference.csv')

    wsis = []
    for idx, wsi_path in enumerate(glob.iglob(os.path.join(wis_img_dir, '**', '*.tif'), recursive=True)):

        # self.wsi_filenames.append(i)
        # print(wsi_path, label_fn(wsi_path))
        # wsis.append(
        wsi = WSI(
            wsi_path,
            mask_path(wsi_path),
            patch_size=512,
            at_mag=5,
            random_rotate=True,
            label_fn=label_fn,
        )

        if wsi.num_patches > 0:
            wsis.append(wsi)

        else:
            print(wsi_path, 'is 0')

        # if idx == 30:
        #     break

    # import sys; sys.exit()
    # print(len(wsis))
    # for wsi in
    dataset = CAMLON16(
        wsis=wsis,
        batch_size=2,
        drop_last=True,
        )

    seq = dataset.patch_len_seq

    dataset.global_seq_len = seq

    # dataset.max_batch_len = 100
    for i in dataset:
         print(i)

path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training'
# mask_dir = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/'

# wis_img_dir = ''

test_camlon16(wis_img_dir=path)