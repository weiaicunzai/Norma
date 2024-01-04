import argparse
import os
import sys
sys.path.append(os.getcwd())

import cv2
import openslide
import pandas
import numpy as np






def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()


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
    # slide_id = 'TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.svs'
    # slide_id = 'TCGA-AR-A1AN-01Z-00-DX1.1118F9FE-6DF2-4496-B102-D3A10D332EC0.svs'
    # import random
    # slide_id = random.choice(os.listdir('/data/hdd1/by/CLAM/tmp/masks')).replace('.png', '.svs')
    # print(slide_id)


    wsi_path = os.path.join(settings.wsi_dir, slide_id)
    mask_path = os.path.join(settings.mask_dir, os.path.splitext(slide_id)[0] + '.png')
    # mask_path = os.path.join('/data/hdd1/by/CLAM/tmp/masks', slide_id.replace('.svs', '.png'))
    print(wsi_path)

    wsi = openslide.OpenSlide(wsi_path)
    level = 0
    dims = None
    for l, d in enumerate(wsi.level_dimensions):
            level = l
            dims = d

    print('read img from wsi {} at level {} dim {}'.format(wsi_path, level, dims))
    img = wsi.read_region((0,0), level, dims).convert('RGB')

    print('read mask from {}'.format(mask_path))
    print(mask_path)
    mask = cv2.imread(mask_path)
    img = np.array(img)
    print(mask.shape, img.shape)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST )
    open_cv_img = img[:, :, ::-1].copy()


    alpha = 0.7
    img = cv2.addWeighted(img, alpha, mask, 1 - alpha, 0)

    cv2.imwrite('tmp/mask.jpg', img)
