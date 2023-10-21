
import os
import sys
import glob

import cv2

from .wsi import WSI
from .camlon16 import CAMLON16Lable





def camlon16_data(wsi_filenames, label_fn):

    def mask_path(wsi_path):
        mask_path = wsi_path.replace('training', 'training_mask')
        mask_path = mask_path.replace('.tif', '.png')
        return mask_path

    wsis = []
    #for i in glob.iglob(os.path.join())
    for wsi_path in wsi_filenames:

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

            print(wsi_path)
            if wsi.num_patches > 0:
                wsis.append(wsi)

            else:
                print(wsi_path, 'is 0')

            # outputs.append(wsi)

    # return outputs
    return wsis
