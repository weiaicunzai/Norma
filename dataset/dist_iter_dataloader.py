import torch

import glob
import os
# from . wsi import

from wsi_reader import camlon16_data
from conf.camlon16 import camlon16_label_fn


class DistWSIDataLoader:
    # DistributedSampler does not support iterabledataset
    def __init__(self,  wis_img_dir, batch_size, num_gpus, num_workers=4, dataset_cls=None, dist=None):
        # self.data_set = data_set

        self.wsis = camlon16_data(wis_img_dir, label_fn=camlon16_label_fn)


        # for idx, wsi_path in enumerate(glob.iglob(os.path.join(wis_img_dir, '**', '*.tif'), recursive=True)):

        #       # self.wsi_filenames.append(i)
        #       # print(wsi_path, label_fn(wsi_path))
        #       # wsis.append(
        #       wsi = WSI(
        #           wsi_path,
        #           mask_path(wsi_path),
        #           patch_size=512,
        #           at_mag=5,
        #           random_rotate=True,
        #           label_fn=label_fn,
        #       )

        #       if wsi.num_patches > 0:
        #           wsis.append(wsi)

        #       else:
        #           print(wsi_path, 'is 0')
