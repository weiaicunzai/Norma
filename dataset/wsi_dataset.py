import random
import warnings
# from itertools import cycle
from utils.mics import cycle
from torch.utils.data import IterableDataset
import torch
import cv2
# default_collate

from PIL import Image



class WSIDataset(IterableDataset):
    # def __init__(self, wsi_img_dir, wsi_mask_dir, patch_size=256, at_mag=20, random_rotate=False):
    # def __init__(self, wsis, batch_size, drop_last, shuffle, transforms=None):
    # def __init__(self, wsis, batch_size, drop_last, transforms=None):
    def __init__(self, wsis, batch_size, transforms=None):
        """the num_worker of each CAMLON16 dataset is one, """
        # self.wis_img_dir =  wsi_img_dir
        # self.wsi_mask_dir = wsi_mask_dir
        self.wsis = wsis
        # self.patch_size = patch_size

        # self.wsi_filenams = []
        # self.wsis = []
        self.batch_size = batch_size
        # self.drop_last = drop_last
        # self.shuffle = shuffle
        assert len(self.wsis) % self.batch_size == 0


        # self.patch_seq = self.get_patch_seq()
        # self.global_seq_len = self.patch_seq
        self.global_seq_len = self.cal_seq_len()
        self.trans = transforms


    def shuffle(self):
        """manually shuffle all the wsis, because when num_workers > 0,
        the copy of dataset wont update in the main process """
                # if self.shuffle:
        random.shuffle(self.wsis)
            # self.global_seq_len = self.get_global_seq_len()

    # @property
    def cal_seq_len(self):
    # def global_seq_len(self):
        outputs = []
        for wsi in self.wsis:
            outputs.append(wsi.num_patches)

        # print('cccccc', outputs, id(self))
        return outputs


        # self.patch_size

        # print(self.wis_img_path)
        # for wsi_path in glob.iglob(os.path.join(self.wis_img_dir, '**', '*.tif'), recursive=True):

        #     # self.wsi_filenames.append(i)
        #     self.wsis.append(WSI(
        #         wsi_path,
        #         self.mask_path(wsi_path),
        #         patch_size=self.patch_size,
        #         at_mag=at_mag,
        #         random_rotate=random_rotate
        #     ))
            # print(i)
            # print(self.mask_path(i))

        # self.start_idx = 0
        # self.end_idx = len(self.wsi) - 1


        # self.end_of_wsi = True # indicate whether the end of a wsi
        # self.patch_scale = 0

        # self.wsi = None
        # self.mask = None
        # self.scale = None
        # self.coordinates = None

        # self.row_idx = 0
        # self.col_idx = 0


    # def mask_path(self, wsi_path):
    #     mask_path = wsi_path.replace('training', 'training_mask')
    #     mask_path = mask_path.replace('.tif', '.png')
    #     return mask_path

    # def level_0_magnification(self, wsi):
    #     if 'aperio.AppMag' in wsi.properties.keys():
    #         level_0_magnification = int(wsi.properties['aperio.AppMag'])
    #     elif 'openslide.mpp-x' in wsi.properties.keys():
    #         level_0_magnification = 40 if int(float(wsi.properties['openslide.mpp-x']) * 10) == 2 else 20
    #     else:
    #         level_0_magnification = 40

    #     return level_0_magnification


#    def get_patch_scale(self, wsi, mask):
#         # if is the start of a new wsi
#         # if self.end_of_wsi:
#             # start compute patch_scale

#             # get the wsi magnifiaction factor of dimension
#             level_0_mag = self.level_0_magnification(wsi)

#             # sample at 20x level
#             if level_0_mag == 40:
#                 level = 1
#             else:
#                 level = 0

#             # print(level)
#             (width, height) =  wsi.level_dimensions[level]
#             mag_20_resolution = (height, width)
#             # print(mag_20_resolution)
#             print(mag_20_resolution, mask.shape)

#             # use multiply to avoid float number comparison
#             assert mag_20_resolution[0] * mask.shape[1] == mag_20_resolution[1] * mask.shape[0]

#             scale = mag_20_resolution[0] / mask.shape[0]

#             return scale



    # def __len__(self):
    #     return len(self.wsi_fns)


    # def get_coordinates(self, row, col, stride):
    #     coords = queue.Queue()

    #     for r_idx in range(0, row, stride):
    #         for c_idx in range(0, col, stride):
    #             coords.put((r_idx, c_idx))

    #     return coords

    # def init_wsi_variables(self, wsi, mask):

    #     self.scale = self.get_patch_scale(self.wsi, self.mask)
    #     # scaled_patch_size = self.patch_size / self.scale

    #     row, col = self.mask.shape[:2]

    #     # self.coords = self.get_coordinates()

    #     self.scale = self.get_patch_scale(self.wsi, self.mask)
    #     stride = int(self.patch_size / self.scale)
    #     # print(stride)
    #     self.coords = self.get_coordinates(row, col, stride)

    # @property
    # def patch_len_seq(self):
    #     num_patch_seq = []

    #     for idx in range(0, len(self.wsis), self.batch_size):

    #         batch_wsi = self.wsis[idx : idx + self.batch_size]

    #         if self.drop_last:
    #             if len(batch_wsi) != self.batch_size:
    #                 return
    #         else:
    #             for i in range(len(batch_wsi), self.batch_size):
    #                 batch_wsi.append(self.wsis[i])

    #         num_patch_seq.append(
    #                 max([wsi.num_patches for wsi in batch_wsi])
    #             )

    #     return num_patch_seq

    # def __getitem__(self, idx):
    def __iter__(self):

        for idx in range(0, len(self.wsis), self.batch_size):

            batch_wsi = self.wsis[idx : idx + self.batch_size]


            # if self.drop_last:
            #     if len(batch_wsi) != self.batch_size:
            #         return
            # else:
            #     for i in range(len(batch_wsi), self.batch_size):
            #     # for i in range(len(batch_wsi), self.start_idx - self.end_idx):
            #         batch_wsi.append(self.wsis[i])


            assert len(batch_wsi) == self.batch_size

            # max_batch_lenth = max([len(x) for x in batch_wsi])

            # print([x.wsi_name for x in batch_wsi])
            batch_wsi = [cycle(x) for x in batch_wsi]
            # for i in range(self.max_batch_len):

            max_len_idx = idx // self.batch_size
            if not self.global_seq_len[max_len_idx]:
                warnings.warn('max batch len equals 0')
                continue
            # for _ in  range(self.global_seq_len[idx // self.batch_size]):
            max_len = self.global_seq_len[max_len_idx]
            for patch_id in  range(max_len):
                # print('ccccccccccc', i)
                # sleep_time = 0.005 * len(batch_wsi)
                # time.sleep(sleep_time)

                # try:

                    # print(batch_wsi)
                    # print(1111, self.global_seq_len[idx // self.batch_size], idx // self.batch_size, _)
                    # outputs = [next(x) for x in batch_wsi]
                    outputs = []
                    for x in batch_wsi:
                        data = next(x).copy()
                        # data = {
                            # 'img': cv2.imread('test_512_patch.jpg'),
                            # 'img': Image.open('test_512_patch.jpg'),
                            # 'label': 1
                        # }
                        # print(type(data['img']))
                        if self.trans is not None:
                            data['img'] = self.trans(data['img'])

                        # data = {
                        #     'img': torch.randn((3, 256, 256)),
                        #     'label': torch.randn((3, 256, 256)),
                        # }
                        if patch_id < max_len - 1:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        # print(patch_id, max_len - 1, data['is_last'])

                        outputs.append(data)

                    # if self.trans is not None:
                        # for data in outputs:
                            # data['img'] = self.trans(x)
                        # outputs = [self.trans(x) for x in outputs]
                    # print()
                    # yield default_collate(outputs)
                    yield outputs

                # except Exception as e:
                    # print('sss', e)

        # if self.shuffle:
        #     random.shuffle(self.wsis)
        #     self.global_seq_len = self.get_global_seq_len()
