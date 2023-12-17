import sys
import os
import re
import glob
import xml.etree.ElementTree as ET

sys.path.append(os.getcwd())

import random
import cv2
import numpy as np
import openslide
import lmdb

from dataset import WSI, WSILMDB
from conf import camlon16
# from dataset import CAMLON16Lable

# from itertools import cycle



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

def read_wsi(wsi_path):

    print(wsi_path)
    open_wsi = openslide.OpenSlide(wsi_path)
    # open_wsi.read_region()

    level = 4
    dims = open_wsi.level_dimensions[level]
    downsample_factor = open_wsi.level_downsamples[level]
    img = open_wsi.read_region((0,0), level, dims)
    # print(img.size)

    return np.array(img), level, downsample_factor


def draw_patch_seq(img, patch_id, cnt, downsample_factor):
    # patch_id = data['patch_id']
    x, y = re.search(r"([0-9]+)_([0-9]+)_1_512_512", patch_id).groups()
    x = int(x)
    y = int(y)
    start = int(x / downsample_factor), int(y / downsample_factor)
    end = int((x + 512 * 2) / downsample_factor), int((y + 512 * 2) / downsample_factor)
    cv2.rectangle(img, start, end, color=(255,0,0), thickness=2)
    text_start = start[0], start[1] + int(1024 / downsample_factor / 1.5)
    cv2.putText(img, str(cnt), text_start, cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=(255,0,0), thickness=2, bottomLeftOrigin=False)

    return img

def parse_anno(xml_path):
    annos = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # print(root)
    # print(xml_path)
    for anno in root.iter('Annotation'):
        res = []
        for coord in anno.iter('Coordinate'):
            res.append([float(coord.attrib['X']), float(coord.attrib['Y'])])

        annos.append(res)
    return annos

def draw_anno(img, anno_path, downsample_factor):
    annos = parse_anno(anno_path)

    tmp = []
    for region in annos:
        region = np.array(region)
        region = region.reshape(-1, 1, 2)
        region = region / downsample_factor
        tmp.append(region.astype(np.int32))


    # template = np.zeros(dims[::-1])
    img = cv2.fillPoly(img, tmp, color=255)
    return img


def draw_patch_seq_with_patch_label(img, patch_id, patch_label, cnt, downsample_factor):
    # patch_id = data['patch_id']
    x, y = re.search(r"([0-9]+)_([0-9]+)_1_512_512", patch_id).groups()
    x = int(x)
    y = int(y)
    start = int(x / downsample_factor), int(y / downsample_factor)
    end = int((x + 512 * 2) / downsample_factor), int((y + 512 * 2) / downsample_factor)
    blue = (255, 0, 0)
    red = (0, 0, 255)
    if patch_label == 1:
        color = red
    else:
        color = blue

    cv2.rectangle(img, start, end, color=color, thickness=2)
    text_start = start[0], start[1] + int(1024 / downsample_factor / 1.5)
    cv2.putText(img, str(cnt), text_start, cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=color, thickness=2, bottomLeftOrigin=False)

    return img


def test_wsirepeat():

    # data_set = 'train'
    data_set = 'test'

    if data_set == 'train':
        dirs = camlon16.train_dirs
        # patch_dir = camlon16.train_dirs
    else:
        dirs = camlon16.test_dirs

    lmdb_path = dirs['lmdb'][0]

    # env = lmdb.open(db_path, readonly=True)

    # count = 0
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    # wsis = []
    # jsons = []

    if data_set == 'train':
        count = random.randint(0, 270)
    else:
        count = random.randint(0, 100)
    # count = 179
    wsi = None
    wsi_path = None
    # count = 0
    # count = 243
    count = 33
    jjj = -1
    direction = 1
    for json_dir, wsi_dir in zip(dirs['jsons'], dirs['wsis']):

        for json_idx, json_path in enumerate(glob.iglob(os.path.join(json_dir, '**', '*.json'), recursive=True)):
            jjj += 1
            if count == jjj:
                wsi = WSILMDB(
                    json_path,
                    patch_json_dir=dirs['patch_level'][0],
                    env=env,
                    direction=direction
                )

                # break
                basename = os.path.basename(json_path).replace('json', 'tif')
                wsi_path = os.path.join(wsi_dir, basename)
                print(wsi_path, json_path, count)

    print(wsi)
    c = [wsi, wsi, wsi]

    count = 0
    # wsi
    for data1, data2, data3 in zip(*c):
        count += 1
        # print(data1['img'], data2, data3)
        # tmp = data1['img'].data_ptr() - data2['img']
        tmp = id(data1['img'])
        tmp1 = id(data2['img'])
        print(tmp, tmp1)
        # , data3)

    # for dd in wsi:
    #     count += 1

    print(count)

def test_wsilmdb():


    data_set = 'train'
    # data_set = 'test'

    if data_set == 'train':
        dirs = camlon16.train_dirs
        # patch_dir = camlon16.train_dirs
    else:
        dirs = camlon16.test_dirs

    lmdb_path = dirs['lmdb'][0]

    # env = lmdb.open(db_path, readonly=True)

    # count = 0
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    # wsis = []
    # jsons = []

    if data_set == 'train':
        count = random.randint(0, 270)
    else:
        count = random.randint(0, 100)
    # count = 179
    wsi = None
    wsi_path = None
    # count = 0
    # count = 243
    jjj = -1
    direction = 1
    cc = 0
    for json_dir, wsi_dir in zip(dirs['jsons'], dirs['wsis']):

        for json_idx, json_path in enumerate(glob.iglob(os.path.join(json_dir, '**', '*.json'), recursive=True)):
            # jjj += 1
            # if count == jjj:
                wsi = WSILMDB(
                    json_path,
                    patch_json_dir=dirs['patch_level'][0],
                    env=env,
                    direction=direction
                )

                # break
                basename = os.path.basename(json_path).replace('json', 'tif')
                wsi_path = os.path.join(wsi_dir, basename)
                # print(wsi_path, json_path, count)
                print(wsi.num_patches)
                cc += wsi.num_patches
                jjj += 1

                # break

    # print('ccccccccc')
    # print(cc / jjj, 'ccc')
    import sys; sys.exit()
    # import sys; sys.exit()


    from viztracer import VizTracer
    tracer = VizTracer()
    # tracer.start()

    img, level, downsample_factor = read_wsi(wsi_path)
    img = np.array(img)


    xml_file = os.path.basename(wsi_path).replace('tif', 'xml')
    xml_path = os.path.join(dirs['anno'][0], xml_file)
    if os.path.exists(xml_path):
        img = draw_anno(img, anno_path=xml_path, downsample_factor=downsample_factor)


    downsample_factor = int(downsample_factor)
    direction = 4
    wsi.direction = direction
    for cnt, data in enumerate(wsi):
        # print(cnt)



        # print(
        assert data['p_label'] == data['label']
        # img = draw_patch_seq(img, data['patch_id'], cnt, downsample_factor)

        # test patch label
        # img = draw_patch_seq_with_patch_label(img, data['patch_id'],  data['p_label'], cnt, downsample_factor)

        # if cnt == 1:
            # print(data['label'], 'ccc', data['p_label'])


    # cv2.imwrite('tmp/{}_test.jpg'.format(direction), img)
    # tracer.stop()
    # tracer.save('results_wsidataloader11.json')


def test_wsi():

    #wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_062.tif'
    #mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_062.png'

    wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/tumor/tumor_006.tif'
    # mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/tumor/tumor_006.png'
    json_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_json/tumor/patch_size_512_at_mag_5/tumor_006.json'

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

    # label_fn = CAMLON16Lable(csv_file='/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/reference.csv')
    # wsi = WSI(wsi_path, mask_path, patch_size=256, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=1024, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=4096, at_mag=5, random_rotate=True, label=label_fn(wsi_path))
    # print(wsi.num_patches, 'num_patches')
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    wsi = WSI(wsi_path, json_path, direction=-1)

    # level_dim = wsi.wsi.level_dimensions[6]
    # a = wsi.wsi.read_region((0, 0), 6, level_dim).convert('RGB')
    # a.save('tmp1/org.jpg')

    print('..................')
    # sus:  /data/ssd1/by/CAMELYON16/training/tumor/tumor_088.tif
    # print(wsi.is_last)
    # for idx, out in enumerate(cycle(wsi)):
    import time
    start = time.time()
    count = 0
    # wsi.construct_random_grids_m(1)
    label = None

    for idx, out in enumerate(wsi):

        if label is None:
            label = out['label']

        assert label == out['label']
        print(out['patch_id'])
        # print(i)
        # img.conv
        # print(wsi.is_last)
        # print(out)
        # img = out['img']
        # out['img'].save('tmp1/{}.jpg'.format(idx))
        count += 1

    end = time.time()

    print((end - start) / count)
    # import sys; sys.exit()

# print(wsi.is_last)
# for idx, img in enumerate(wsi):
#     # print(i)
#     # img.conv
#     print(wsi.is_last)
#     img.save('tmp1/{}.jpg'.format(idx))

# print(wsi.num_patches)
# print(wsi.is_last)


# test_num_patches()
# test_wsi()
test_wsilmdb()
# test_wsirepeat()