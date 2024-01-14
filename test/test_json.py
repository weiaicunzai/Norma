import argparse
import sys
import os
import re
import glob
import xml.etree.ElementTree as ET
import json

sys.path.append(os.getcwd())

import random
import cv2
import numpy as np
import openslide
import lmdb
import pandas

from dataset import WSI, WSILMDB
from conf import camlon16
import xml.etree.ElementTree as ET


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
# test_wsilmdb()
# test_wsirepeat()

def draw_patch_seq(img, patch_id, cnt, level_dims, level_factor):
    # patch_id = data['patch_id']
    # x, y = re.search(r"([0-9]+)_([0-9]+)_1_512_512", patch_id).groups()
    # x = patch_id[0]
    # y = patch_id[1]
    # patch_size = patch_id
    (x, y), level, (patch_size, patch_size) = patch_id # x, y: top_level x, y
    # print(patch_id)
    scaled_factor = level_factor[level]
    dims = level_dims[0] # x, y
    img_height, img_width = img.shape[:2]
    downsample_factor_x = dims[0] / img_width
    # print(downsample_factor, dims, img.shape[:2])
    downsample_factor_y = dims[1] / img_height
    # print(dims, img.shape[:2])

    assert abs(dims[0] / img_width - dims[1] / img_height) < 1
    # x = int(x)
    # y = int(y)
    # start = int(x / downsample_factor), int(y / downsample_factor)
    start = int(x / downsample_factor_x), int(y / downsample_factor_y)
    # end = int((x + 512 * 2) / downsample_factor), int((y + 512 * 2) / downsample_factor)
    # end = int((x + patch_size) / downsample_factor), int((y + patch_size) / downsample_factor)
    end = int((x + patch_size * scaled_factor) / downsample_factor_x), int((y + patch_size * scaled_factor) / downsample_factor_y)
    # print(start, end, img.shape)
    cv2.rectangle(img, start, end, color=(255,0,0), thickness=2)
    # text_start = start[0], start[1] + int(1024 / downsample_factor / 1.5)
    # text_start = start[0], start[1] + int(patch_size / downsample_factor / 1.5)
    # bot_left corrner
    text_start = start[0] + int(patch_size * scaled_factor / downsample_factor_x / 20), start[1] + int(patch_size * scaled_factor / downsample_factor_y / 1.5)
    # text_start = start[0], start[1] + int(patch_size / downsample_factor)
    cv2.putText(img, str(cnt), text_start, cv2.FONT_HERSHEY_COMPLEX, fontScale=0.25, color=(255,0,0), thickness=1, bottomLeftOrigin=False)

    return img

def vis_mask(img, mask_path):
    # wsi = openslide.OpenSlide(wsi_path)

    # seg_level = len(wsi.level_dimensions) - 7
    # # seg_level = -2
    # img = wsi.read_region((0,0), seg_level, wsi.level_dimensions[seg_level]).convert('RGB')
    # img = np.array(img)

    mask = cv2.imread(mask_path, -1)
    print(img.shape, mask.shape)

    img = cv2.resize(img, mask.shape[::-1])

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    res = cv2.addWeighted(img, 0.7, mask, 0.3, 0)

    return res

def draw_cam16_anno(img, slide_id, settings, dims):

    name = os.path.splitext(slide_id)[0]
    xml_path = os.path.join(settings.anno_dir, name + '.xml')

    if not os.path.exists(xml_path):
        return img

    coords = parse_anno(xml_path)
    downsample_factor = dims[1] / img.shape[0]

    tmp = []
    for idx, region in enumerate(coords):
        region = np.array(region)
        region = region.reshape(-1, 1, 2)
        region = region / downsample_factor
        tmp.append(region.astype(np.int32))
        # if idx > 2:
            # break

    print('total {} areas of tumor'.format(len(tmp)))
    # img = cv2.polylines(cv_img, tmp, False, (0, 255, 255))
    img = cv2.polylines(img, tmp, False, (0, 255, 255), thickness=2)
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



def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()

# def draw_patch_id(seg_path, )

if __name__  == '__main__':
    args = get_args_parser()
    if args.dataset == 'brac':
        from conf.brac import settings
    elif args.dataset == 'cam16':
        from conf.camlon16 import settings
    else:
        raise ValueError('wrong value error')

    df = pandas.read_csv(settings.file_list_csv)
    row = df.sample(n=1)
    slide_id = row.iloc[0]['slide_id']
    # tumor_043.tif only has one patch is wrong

    wsi_path = os.path.join(settings.wsi_dir, slide_id)
    mask_path = os.path.join(settings.mask_dir, os.path.splitext(slide_id)[0] + '.png')

    wsi = openslide.OpenSlide(wsi_path)

    for seg_level in range(len(wsi.level_dimensions) - 1, -1, -1):
        if max(wsi.level_dimensions[seg_level]) > 10000:
            break


    seg_level = len(wsi.level_dimensions) - 6
    # seg_level = -2
    img = wsi.read_region((0,0), seg_level, wsi.level_dimensions[seg_level]).convert('RGB')
    img = np.array(img)

    img = vis_mask(img, mask_path)

    json_path = os.path.join(settings.json_dir, os.path.splitext(slide_id)[0] + '.json')
    direction = random.choice(range(8))
    direction = 5

    msg =   {
                0: 'row         + col          + row first',
                1: 'row         + col          + col first',
                2: 'row         + revserse col + row first',
                3: 'row         + revserse col + col first',
                4: 'reverse row + col          + row first',
                5: 'reverse row + col          + col first',
                6: 'reverse row + reverse col  + row first',
                7: 'reverse row + reverse col  + col first',
            }

    print('direction is {}, it should be in order {}'.format(direction, msg[direction]))

    json_data = json.load(open(json_path, 'r'))

    # print(json_data.keys())
    print('file name {}'.format(json_data['filename']))
    print('label {}'.format(json_data['label']))
    coords = json_data['coords'][direction]
    # print(len(json_data['coords']))
    # print(len(coords))
    # print(json_path)
    # for k, v in json_data.items():
        # print(k)
    print('total {} number of patches'.format(len(coords)))

    wsi = openslide.OpenSlide(wsi_path)
    level_dims = wsi.level_dimensions
    level_factor = wsi.level_downsamples

    # incase the last level is too small
    # writing text would be hard to read
    img = cv2.resize(img, (0, 0), fx=3, fy=3)

    for idx, patch_id in enumerate(coords):
        # print(patch_id)
        # print(patch_id)
        img = draw_patch_seq(img, patch_id, idx, level_dims, level_factor)

# def draw_cam16_anno(img, slide_id, settings, dims):

    if args.dataset == 'cam16':
        img = draw_cam16_anno(img, slide_id, settings, level_dims[0])
    cv2.imwrite('tmp/img_test_json.jpg', img)