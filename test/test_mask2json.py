import sys
import os

sys.path.append(os.getcwd())
from dataset import WSI
from dataset import CAMLON16Lable


from preprocess.mask2json import MaskConverter



def test_num_patches():

    wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_063.tif'
    mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_063.png'


    # wsi = WSI(wsi_path, mask_path, patch_size=4096, at_mag=5, random_rotate=True)
    # wsi = WSI(wsi_path, mask_path, patch_size=4096, at_mag=20, random_rotate=True)
    wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, fg_thresh=0.33)
    #wsi = WSI(wsi_path, mask_path, patch_size=1024, at_mag=5, random_rotate=True)
    print(wsi.num_patches)
# print(wsi)
# wsi_path, mask_path, patch_size, at_mag=20

# print(wsi.is_end)

def test_wsi():

    #wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_062.tif'
    #mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_062.png'

    wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/tumor/tumor_006.tif'
    mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/tumor/tumor_006.png'

    #wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/tumor/tumor_075.tif'
    #mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/tumor/tumor_075.png'

    #wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/tumor/tumor_045.tif'
    #mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/tumor/tumor_045.png'

    #wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_045.tif'
    #mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_045.png'
    # mask_path = 'tmp1/mask.png'

    #wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_045.tif'
    #mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_045.png'

    #wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_042.tif'
    #mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_042.png'

    # wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_027.tif'
    # mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_027.png'

    label_fn = CAMLON16Lable(csv_file='/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/reference.csv')
    # wsi = WSI(wsi_path, mask_path, patch_size=256, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=1024, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = MaskConverter(wsi_path, mask_path, patch_size=2048, at_mag=5, random_rotate=True, label_fn=label_fn)
    wsi = MaskConverter(wsi_path, mask_path, patch_size=4096, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi.construct_random_grids_m(5)
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=4096, at_mag=5, random_rotate=True, label=label_fn(wsi_path))
    # print(wsi.num_patches, 'num_patches')

    # level_dim = wsi.wsi.level_dimensions[6]
    # a = wsi.wsi.read_region((0, 0), 6, level_dim).convert('RGB')
    # a.save('tmp1/org.jpg')

    print('..................')
    # print(wsi.is_last)
    # for idx, out in enumerate(cycle(wsi)):
    import time
    start = time.time()
    count = 0
    print(wsi.num_patches)
    for i in range(8):
        wsi.construct_random_grids_m(i)
        for idx, out in enumerate(wsi):
            # print(i)
            # img.conv
            # print(wsi.is_last)
            print(out)
            # out['img'].save('tmp1/{}.jpg'.format(idx))
            count += 1

    end = time.time()

    print((end - start) / count)

# print(wsi.is_last)
# for idx, img in enumerate(wsi):
#     # print(i)
#     # img.conv
#     print(wsi.is_last)
#     img.save('tmp1/{}.jpg'.format(idx))

# print(wsi.num_patches)
# print(wsi.is_last)


# test_num_patches()
test_wsi()