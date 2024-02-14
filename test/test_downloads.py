import argparse
import os
import sys
import glob
import random
import csv
sys.path.append(os.getcwd())
import multiprocessing

import openslide

def get_wsi_path(settings):
    csv_file = settings.file_list_csv
    with open(csv_file, 'r') as f:
        for row in csv.DictReader(f):
            yield os.path.join(settings.wsi_dir, row['slide_id'])

def is_broken(wsi_path):
    print('checking file {}'.format(wsi_path))
    if not os.path.exists(wsi_path):
        print('file {} do not exists'.format(wsi_path))
        return True

    wsi = openslide.OpenSlide(wsi_path)

    for _ in range(30):
        level = random.randint(0,len(wsi.level_dimensions)-1)
        d1, d2 = wsi.level_dimensions[level]
        d1 = random.randint(0, d1 - 256)
        d2 = random.randint(0, d2 - 256)

        try:
            c = wsi.read_region((d1, d2), level, (512, 512))
        except Exception as e:
            print((d1, d2), level, (512, 512))
            print(wsi)
            print(e)
            count += 1
            # break
            return True

    return False

def random_read(settings):
    pool = multiprocessing.Pool(processes=4)
    res = pool.map(is_broken, get_wsi_path(settings))

    print('total {} files are broken'.format(sum(res)))

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()

if __name__  == '__main__':
    args = get_args_parser()
    if args.dataset == 'brac':
        from conf.brac import settings
    elif args.dataset == 'bracs':
        from conf.bracs import settings
    elif args.dataset == 'lung':
        from conf.lung import settings
    else:
        raise ValueError('wrong value error')

    random_read(settings)