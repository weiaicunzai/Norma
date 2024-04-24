import os
import sys
import argparse

sys.path.append(os.getcwd())
import glob
import json
import time
import csv
import multiprocessing as mp
from functools import partial

import cv2
import numpy as np
import openslide

class MaskConverter:
    def __init__(self, wsi_path, mask_path, patch_size, at_mag=20, fg_thresh=0.33):

        """at_mag: extract patches at magnification xxx"""

        assert at_mag in [20, 40, 10, 5]
        self.wsi = openslide.OpenSlide(wsi_path)

        self.level_0_mag = self.get_level_0_mag()

        self.mask = cv2.imread(mask_path, -1)
        print(mask_path)
        cv2.imwrite('masks.jpg', self.mask)

        self.mask_scale = self.get_mask_scale()

        if self.mask is None:
            raise ValueError('{} is none'.format(mask_path))

        self.patch_size = int(patch_size)  # extracted patch resolution at "at_mag"
        self.at_mag = int(at_mag)

        self.kernel_size = self.get_kernel_size()


        # if current patch is the last patch
        self.fg_thresh = fg_thresh

        self.assert_mag()
        self.num_patches = self.cal_num_patches()

    def assert_mag(self):
        self.level_0_mag
        for downsample_factor in self.wsi.level_downsamples:
            # downsample_factor
            mag = round(self.level_0_mag / downsample_factor)
            if self.at_mag == mag:
                return

        raise ValueError('wsi {} do not have mag {}'.format(self.wsi, self.at_mag))




    def get_kernel_size(self):
         level = self.mag2level(self.at_mag)
         scale = self.wsi.level_downsamples[level]
         # resized self.patch_size to level 0, then divide self.mask_scale
         kernel_size = int(self.patch_size * scale / self.mask_scale)

         return kernel_size

    def cal_num_patches(self):

        """calculate number of patches by average pooling of each mask, and the average value
        larger than given self.fg_thresh * mask.max()
        """
        row_num, col_num = self.mask.shape[:2]

        # print('mask', self.mask.mean())
        # cv2.imwrite('masks.jpg', self.mask)


        kernel_size = self.kernel_size

        pad_row = kernel_size - row_num % kernel_size
        pad_col = kernel_size - col_num % kernel_size


        padded_mask = np.pad(self.mask, ((0, pad_row), (0, pad_col)), mode='constant', constant_values=(0, 0))

        # cv2.imwrite('pad.jpg', padded_mask)

        padded_row, padded_col = padded_mask.shape[:2]

        assert padded_row % kernel_size == 0
        assert padded_col % kernel_size == 0

        assert padded_mask.ndim == 2

        reshape_mask = padded_mask.reshape(
            (
                padded_row // kernel_size,
                kernel_size,
                padded_col // kernel_size,
                kernel_size
            )
        )

        mean_values = reshape_mask.mean(axis=(1, 3))
        mean_mask = mean_values > reshape_mask.max() * self.fg_thresh
        # print(mean_mask.mean())
        # print(mean_values)

        return mean_mask.sum()


    def mag2level(self, mag):

        level_0_mag = self.level_0_mag
        assert level_0_mag >= mag
        assert mag in [20, 40, 10, 5]

        tmp_level = level_0_mag
        levels = []
        while True:
            if tmp_level < 5:
                break

            levels.append(tmp_level)
            tmp_level = int(tmp_level / 2)

        for level_idx, level_mag in enumerate(levels):
            #print(level_idx, level_mag)
            if level_mag == mag:
                return level_idx

        raise ValueError('something wrong????')


    # @property
    def get_level_0_mag(self):
        if 'aperio.AppMag' in self.wsi.properties.keys():
            level_0_magnification = int(float(self.wsi.properties['aperio.AppMag']))
        elif 'openslide.mpp-x' in self.wsi.properties.keys():
            level_0_magnification = 40 if int(float(self.wsi.properties['openslide.mpp-x']) * 10) == 2 else 20
        else:
            if max(self.wsi.level_dimensions[0]) < 50000:
                level_0_magnification = 20
            else:
                level_0_magnification = 40

        return level_0_magnification

    # @property
    def get_mask_scale(self):
        """get scale factor of mask compared to level 0 of wsi"""

        # level 0 dim
        (width, height) =  self.wsi.level_dimensions[0] # fixed

        level_0_resolution = (height, width)  # fixed

        # use multiply to avoid float number comparison
        # assert mag_20_resolution[0] * self.mask.shape[1] == mag_20_resolution[1] * self.mask.shape[0]
        # print(self.mask.shape)
        # print(self.wsi.level_dimensions)
        # assert level_0_resolution[0] * self.mask.shape[1] == level_0_resolution[1] * self.mask.shape[0]

        # scale = mag_20_resolution[0] / self.mask.shape[0]
        # level_0_resolution[0] / self.mask.shape[0] and level_0_resolution[1] / self.mask.shape[1]
        # are not neccesary the same, however, thus small erorr does not effect our patching results
        scale = round(level_0_resolution[0] / self.mask.shape[0])

        return scale


    def construct_random_grids_m(self, direction):
        """rotate the whole slide image 90, 180, 270 degree clockwise
        or counter clockwise. However, since the resolution of WSI is
        large, we simulate WSI rotations by extracting patches start
        from different directions
        note: flip WSI as data augmentation is unnecessary since the patches
        will be random flipped.
        0: row         + col          + row first           (normal)
        1: row         + col          + col first
        2: row         + revserse col + row first
        3: row         + revserse col + col first
        4: reverse row + col          + row first
        5: reverse row + col          + col first
        6: reverse row + reverse col  + row first
        7: reverse row + reverse col  + col first
        """

        assert direction in [0, 1, 2, 3, 4, 5, 6, 7]

        coords = []

        row, col = self.mask.shape[:2]

        stride = self.kernel_size

        row_iter = range(0, int(row), stride)
        # if direction is in [4, 5, 6, 7]
        # reverse row
        if direction in [4, 5, 6, 7]:
            row_iter = sorted(row_iter, reverse=True)

        # if direction is in
        col_iter = range(0, int(col), stride)
        if direction in [2, 3, 6, 7]:
            col_iter = sorted(col_iter, reverse=True)

        # row first
        if direction in [0, 2, 4, 6]:
            for r_idx in row_iter:
                for c_idx in col_iter:
                    coords.append((r_idx, c_idx))
        # col first
        else:
            for c_idx in col_iter:
                for r_idx in row_iter:
                    coords.append((r_idx, c_idx))

        self.grids = coords


    def construct_grids_m(self):
        """
            return patch coords, (r_idx, c_idx) in mask coordinates,

        """
        coords = []

        row, col = self.mask.shape[:2]

        stride = self.kernel_size

        for r_idx in range(0, int(row), stride):
            for c_idx in range(0, int(col), stride):
                coords.append((r_idx, c_idx))

        self.grids = coords

    def __iter__(self):

        for coord_m in self.grids:
            r_idx, c_idx = coord_m

            # convert to level 0 coords
            top_left_xy = (c_idx * self.mask_scale, r_idx * self.mask_scale)
            top_left_xy = [int(x) for x in top_left_xy]

            for x in top_left_xy:
                # print(top_left_xy, self.patch_size, self.mask_scale, c_idx)
                assert x % self.patch_size == 0

            # print(top_left_xy, self.patch_size)
            patch_mask = self.mask[r_idx : r_idx + self.kernel_size, c_idx: c_idx + self.kernel_size]

            # remove patch block extended the image borader
            if patch_mask.shape[0] != self.kernel_size:
                continue

            # remove patch block extended the image borader
            if patch_mask.shape[1] != self.kernel_size:
                continue

            if (patch_mask > 0).sum() / (self.kernel_size * self.kernel_size) < self.fg_thresh:
                continue

            yield top_left_xy, self.mag2level(self.at_mag), (self.patch_size, self.patch_size)

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

def get_real_mag(wsi_path, at_mag):

    wsi = openslide.OpenSlide(wsi_path)

    if 'aperio.AppMag' in wsi.properties.keys():
        level_0_magnification = int(float(wsi.properties['aperio.AppMag']))
    elif 'openslide.mpp-x' in wsi.properties.keys():
        level_0_magnification = 40 if int(float(wsi.properties['openslide.mpp-x']) * 10) == 2 else 20
    else:
        if max(wsi.level_dimensions[0]) < 50000:
            level_0_magnification =  20
        else:
            level_0_magnification = 40

    real_mag = level_0_magnification
    for factor in wsi.level_downsamples:
        # factor can be float
        if level_0_magnification / round(factor) == at_mag:
            real_mag = at_mag

    return real_mag, level_0_magnification


def write_single_json(slide_id, label, settings):

    res = {

    }
    res['filename'] = slide_id
    res['label'] = label
    res['coords'] = []

    wsi_path = os.path.join(settings.wsi_dir, slide_id)
    name = os.path.splitext(slide_id)[0]
    mask_path = os.path.join(settings.mask_dir, name + '.png')

    real_mag, level_0_mag = get_real_mag(wsi_path=wsi_path, at_mag=settings.mag)


    assert real_mag >= settings.mag

    # if given mag is missing, extracted from larger mag
    patch_size = int((real_mag / settings.mag) * settings.patch_size)
    at_mag = real_mag

    wsi = MaskConverter(wsi_path, mask_path, patch_size=patch_size, at_mag=at_mag)

    start = time.time()
    for i in range(8):
        coords = []
        wsi.construct_random_grids_m(i)
        for idx, out in enumerate(wsi):
            coords.append(out)

        res['coords'].append(coords)


    assert len(res['coords'])  == 8

    assert all_equal([len(x) for x in res['coords']])

    wsi_filename = os.path.basename(wsi_path)
    json_filename = os.path.splitext(wsi_filename)[0] + '.json'

    dest_dir = settings.json_dir

    json_save_path = os.path.join(dest_dir, json_filename)
    with open(json_save_path, 'w') as f:
        json.dump(res, f)

    end = time.time()
    print('the orig mag is {}, extracting {} patches at {} with patch_size {} from {}, using time {}, writing to {}'.format(
        level_0_mag, wsi.cal_num_patches(), at_mag, patch_size, wsi_path, end - start, json_save_path))

def get_filenames(settings):

    csv_path = settings.file_list_csv
    with open(csv_path, newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            # if row['slide_id'] != 'TCGA-BH-A2L8-01Z-00-DX1.ACA51CA9-3C38-48A6-B4A9-C12FFAB9AB56.svs':
            #     continue
            if row['slide_id'] != 'test_116.tif':
                continue
            yield row['slide_id'], row['label']

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()

if __name__ == '__main__':

    args = get_args_parser()
    if args.dataset == 'brac':
        from conf.brac import settings
    elif args.dataset == 'cam16':
        from conf.camlon16 import settings
    elif args.dataset == 'lung':
        from conf.lung import settings
    else:
        raise ValueError('wrong dataset')

    dest_dir = settings.json_dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    pool = mp.Pool(processes=mp.cpu_count()) # computation bound operation
    pool.starmap(partial(write_single_json, settings=settings), get_filenames(settings))
    pool.close()