
import os
import sys
sys.path.append(os.getcwd())
import cv2
import openslide

from preprocess.seg_tissue import segment_tissue
# from dataset import WSI, CAMLON16Lable



def test_tissue_segmentation():

    wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_045.tif'
    # mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_045.png'

    # label_fn = CAMLON16Lable(csv_file='/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/reference.csv')
    # wsi = WSI(wsi_path, mask_path, patch_size=256, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=1024, at_mag=5, random_rotate=True, label_fn=label_fn)
    wsi = openslide.OpenSlide(wsi_path)

    # filter_params={'a_t':100 // 8, 'a_h': 16 // 4, 'max_n_holes': 8 // 4}
    filter_params={'a_t':100 // 32, 'a_h': 16 // 4, 'max_n_holes': 8 // 4}
    print(filter_params)
    mask = segment_tissue(wsi,  seg_level=6, use_otsu=True, filter_params=filter_params)
    print(mask.sum())

    cv2.imwrite('tmp1/mask.png', mask)



test_tissue_segmentation()