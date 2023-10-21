
import os
import sys
sys.path.append(os.getcwd())
import cv2
from utils.vis import vis_mask














if __name__ == '__main__':

    wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_045.tif'
    # mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_045.png'
    mask_path = 'tmp1/mask.png'

    out = vis_mask(wsi_path, mask_path, 6)

    cv2.imwrite('test.jpg', out)
