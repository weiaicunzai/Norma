import os
import sys

sys.path.append(os.getcwd())
import glob
import json
import time

import cv2
import numpy as np
import openslide

from dataset.camlon16 import  CAMLON16Lable

class MaskConverter:
    # def __init__(self, wsi_path, mask_path, patch_size, at_mag=20, random_rotate=False, fg_thresh=0.7):
    def __init__(self, wsi_path, mask_path, patch_size, label_fn, at_mag=20, random_rotate=False, fg_thresh=0.33):

        """at_mag: extract patches at magnification xxx"""

        # print(wsi_path, mask_path)

        assert at_mag in [20, 40, 10, 5]
        self.wsi = openslide.OpenSlide(wsi_path)
        # print(self.wsi.level_downsamples)
        self.mask = cv2.imread(mask_path, -1)

        if self.mask is None:
            raise ValueError('{} is none'.format(mask_path))

        self.patch_size = patch_size  # extracted patch resolution at "at_mag"
        self.at_mag = at_mag
        self.random_rotate = random_rotate

        if self.random_rotate == False:
            self.construct_grids_m()
        else:
            self.construct_random_grids_m(0)

        # if current patch is the last patch
        # self.is_last = False
        self.fg_thresh = fg_thresh

        self.num_patches = self.cal_num_patches()

        self.label = label_fn(wsi_path)

        # elif 'normal' in wsi_path:
        #     self.label = 1
        # else:
        #     raise ValueError('wrong value')
        self.wsi_name = os.path.basename(wsi_path)

    @property
    def kernel_size(self):
         level = self.mag2level(self.at_mag)
         scale = self.wsi.level_downsamples[level]
         kernel_size = int(self.patch_size * scale / self.mask_scale)

         return kernel_size

    def cal_num_patches(self):

        """calculate number of patches by average pooling of each mask, and the average value
        larger than given self.fg_thresh * mask.max()
        """
        row_num, col_num = self.mask.shape[:2]
        # kernel_size =

        # level = self.mag2level(self.at_mag)
        # scale = self.wsi.level_downsamples[level]

        # kernel_size = int(self.patch_size * scale / self.mask_scale)
        # print(row_num,  self.kernel_size)
        kernel_size = self.kernel_size

        # print(self.patch_size, 'patch_size')
        pad_row = kernel_size - row_num % kernel_size
        pad_col = kernel_size - col_num % kernel_size
        # print(row_num, col_num, pad_row, pad_col, kernel_size)


        padded_mask = np.pad(self.mask, ((0, pad_row), (0, pad_col)), mode='constant', constant_values=(0, 0))
        # print(padded_mask.shape, self.mask.shape)
        # cv2.imwrite(
        #     'padded_mask.png', padded_mask
        # )
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

        # print(reshape_mask.shape)

        # count = 0
        # for idx in range(reshape_mask.shape[0]):
        #     for idx1 in range(reshape_mask.shape[2]):
        #         # print(idx, idx1)
        #         count += 1
        #         cv2.imwrite('tmp1/mask_{}.png'.format(count), reshape_mask[idx, :, idx1, :])


        mean_values = reshape_mask.mean(axis=(1, 3))
        # print(mean_values, reshape_mask.max())
        #mean_values
        # mean_values[mean_values]
        mean_mask = mean_values > reshape_mask.max() * self.fg_thresh
        # print(mean_values, reshape_mask.max() * self.fg_thresh)
        # print((mean_values > reshape_mask.max() * self.fg_thresh).sum())
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


    # def get_dimension

    # @property
    # def is_end(self):
    #     if len(self.grids):
    #         return False
    #     else:
    #         return True

    @property
    def level_0_mag(self):
        if 'aperio.AppMag' in self.wsi.properties.keys():
            level_0_magnification = int(self.wsi.properties['aperio.AppMag'])
        elif 'openslide.mpp-x' in self.wsi.properties.keys():
            level_0_magnification = 40 if int(float(self.wsi.properties['openslide.mpp-x']) * 10) == 2 else 20
        else:
            level_0_magnification = 40

        return level_0_magnification

    # def convert_to_level
    # def convert_level_dim(dim, src_level, )
    # def level_0_scale(self):
        # level = self.mag2level(self.at_mag)

    # def level0_scale(self, )

    @property
    def mask_scale(self):
        """get scale factor of mask compared to level 0 of wsi"""
        # level = self.mag2level(self.at_mag)
        # level = ``

        # (width, height) =  self.wsi.level_dimensions[level]

        # level 0 dim
        (width, height) =  self.wsi.level_dimensions[0]

        level_0_resolution = (height, width)

        # use multiply to avoid float number comparison
        # assert mag_20_resolution[0] * self.mask.shape[1] == mag_20_resolution[1] * self.mask.shape[0]
        # print(self.mask.shape)
        # print(self.wsi.level_dimensions)
        # assert level_0_resolution[0] * self.mask.shape[1] == level_0_resolution[1] * self.mask.shape[0]

        # scale = mag_20_resolution[0] / self.mask.shape[0]
        # level_0_resolution[0] / self.mask.shape[0] and level_0_resolution[1] / self.mask.shape[1]
        # are not neccesary the same, however, thus small erorr does not effect our patching results
        scale = level_0_resolution[0] / self.mask.shape[0]

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

        # level = self.mag2level(self.at_mag)
        # scale = self.wsi.level_downsamples[level]
        # stride = int(self.patch_size * scale / self.mask_scale)

        stride = self.kernel_size

        # row_iter = random.choice([range(0, int(row), stride), reversed(range(0, int(row), stride)])
        row_iter = range(0, int(row), stride)
        # if random.random() > 0.5:
        if direction in [4, 5, 6, 7]:
        # row_iter = reversed(row_iter)
            row_iter = sorted(row_iter, reverse=True)




        col_iter = range(0, int(col), stride)
        # if random.random() > 0.5:
        if direction in [2, 3, 6, 7]:
            # col_iter = range(0, int(col), stride)
        # col_iter = reversed(col_iter)
            col_iter = sorted(col_iter, reverse=True)


        # if random.random() > 0.5:
        # row first
        # print(direction)
        if direction in [0, 2, 4, 6]:
            for r_idx in row_iter:
                for c_idx in col_iter:
                    coords.append((r_idx, c_idx))

        else:
            # print('ccc')
        # print('col first')
            for c_idx in col_iter:
                # print('col', c_idx)
                for r_idx in row_iter:
                    # print(r_idx, c_idx, 'ccc')
                    coords.append((r_idx, c_idx))


        self.grids = coords


    def construct_grids_m(self):
        """
            return patch coords, (r_idx, c_idx) in mask coordinates,

        """

        # coords = queue.Queue()
        coords = []

        row, col = self.mask.shape[:2]

        # level = self.mag2level(self.at_mag)
        # scale = self.wsi.level_downsamples[level]
        # stride = int(self.patch_size * scale / self.mask_scale)
        stride = self.kernel_size

        for r_idx in range(0, int(row), stride):
            for c_idx in range(0, int(col), stride):
                coords.append((r_idx, c_idx))

        self.grids = coords
        # return coords

    def __iter__(self):

        # self.is_last = False
        # while not self.is_end()
        # while not self.is_end():
        # print(len(self.grids), 'self.grids len')
        count = 0
        for coord_m in self.grids:
            # coords = self.grids.get()
            r_idx, c_idx = coord_m
            # print('colllll', c_idx)

            # convert to level 0 coords
            # level = self.mag2level(self.at_mag)
            # scale = self.wsi.level_downsamples[level]
            # top_left_xy = (c_idx * scale, r_idx * scale)
            top_left_xy = (c_idx * self.mask_scale, r_idx * self.mask_scale)
            # print(top_left_xy)
            top_left_xy = [int(x) for x in top_left_xy]
            # print(top_left_xy, (self.patch_size, self.patch_size))

            patch_mask = self.mask[r_idx : r_idx + self.kernel_size, c_idx: c_idx + self.kernel_size]
            # print((patch_mask > 0).sum() )
            # print(patch_mask)
            # print(r_idx, r_idx + self.kernel_size, c_idx, c_idx + self.kernel_size, self.mask.shape, patch_mask.shape)
            # print(patch_mask.max())
            # print(patch_mask)

            if (patch_mask > 0).sum() / (self.kernel_size * self.kernel_size) < self.fg_thresh:
                continue

            yield top_left_xy, self.mag2level(self.at_mag), (self.patch_size, self.patch_size)
            # wsi.read_region((x, y), level, size)
            # (x, y): the top-left coordinates in level 0
            # level: the level which patches are extracted
            # patch_size, patch_size: the extracted patch size at level "level"
            # count += 1
            # # print(self.wsi_name)
            # assert count <= self.num_patches
            # yield {
            #         'img': self.wsi.read_region(top_left_xy,
            #                        self.mag2level(self.at_mag),
            #                        (self.patch_size, self.patch_size)).convert('RGB'),
            #          'label': self.label,
            #          'num_patches': self.num_patches,
            #          'test': self.wsi_name,
            #          'idx': count,
            #        }
            # yield

        # raise End
        # self.is_last = True


def mask_path(wsi_path):
    if 'training' in wsi_path:
        mask_path = wsi_path.replace('training', 'training_mask')

    if 'testing' in wsi_path:
        mask_path = wsi_path.replace('images', 'masks')
    mask_path = mask_path.replace('.tif', '.png')
    return mask_path

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

def write_json(wsi_dir, dest_dir, patch_size, at_mag):



    label_fn = CAMLON16Lable(csv_file='/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/reference.csv')
    # wsi = WSI(wsi_path, mask_path, patch_size=256, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=1024, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = MaskConverter(wsi_path, mask_path, patch_size=2048, at_mag=5, random_rotate=True, label_fn=label_fn)
    res = {

    }
    for wsi_path in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
        print(wsi_path)

        res['filename'] = os.path.basename(wsi_path)
        res['label'] = label_fn(wsi_path)
        res['coords'] = []
        # print(label_fn())

        # wsi = openslide.OpenSlide(wsi_path)
        wsi = MaskConverter(wsi_path, mask_path(wsi_path), patch_size=patch_size, at_mag=at_mag, random_rotate=True, label_fn=label_fn)
        # wsi.construct_random_grids_m(5)
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=512, at_mag=5, random_rotate=True, label_fn=label_fn)
    # wsi = WSI(wsi_path, mask_path, patch_size=4096, at_mag=5, random_rotate=True, label=label_fn(wsi_path))
        # print(wsi.num_patches, 'num_patches')

        # level_dim = wsi.wsi.level_dimensions[6]
        # a = wsi.wsi.read_region((0, 0), 6, level_dim).convert('RGB')
        # a.save('tmp1/org.jpg')

        print('..................')
        print(wsi.cal_num_patches())
        # print(wsi.is_last)
        # for idx, out in enumerate(cycle(wsi)):
        start = time.time()
        count = 0
        for i in range(8):
            coords = []
            wsi.construct_random_grids_m(i)
            for idx, out in enumerate(wsi):
                # print(i)
                # img.conv
                # print(wsi.is_last)
                # print(out)
                # out['img'].save('tmp1/{}.jpg'.format(idx))
                count += 1
                coords.append(out)

            res['coords'].append(coords)

        end = time.time()

        print((end - start) / count)

        assert len(res['coords'])  == 8

        assert all_equal([len(x) for x in res['coords']])

        wsi_filename = os.path.basename(wsi_path)
        json_filename = wsi_filename.replace('.tif', '.json')

        sub_folder = 'patch_size_{}_at_mag_{}'.format(patch_size, at_mag)

        if not os.path.exists(os.path.join(dest_dir, sub_folder)):
            os.makedirs(os.path.join(dest_dir, sub_folder))
            # print(os.path.join(dest_dir, sub_folder, json_filename))

        print(os.path.join(dest_dir, sub_folder, json_filename))
        with open(os.path.join(dest_dir, sub_folder, json_filename), 'w') as f:
            # f.write(res, f)
            json.dump(res, f)


        # import sys; sys.exit()










if __name__ == '__main__':

# from random import randint

# path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/images/test_{:03d}.tif'.format(randint(1, 100))

    path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/tumor/'
    dest_dir = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_json/tumor/'

    # write_json(path, dest_dir, 512, 5)
    write_json(path, dest_dir, 512, 20)

    path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/'
    dest_dir = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_json/normal/'

    # write_json(path, dest_dir)
    write_json(path, dest_dir, 512, 20)

    path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/images/'
    dest_dir = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/jsons'

    # write_json(path, dest_dir, 512, 5)
    write_json(path, dest_dir, 512, 20)
