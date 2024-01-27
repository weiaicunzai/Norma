import argparse
import csv
import io
import multiprocessing as mp
import json
import os
import time
import sys
from functools import partial
from multiprocessing import Process, Queue, Manager


import lmdb
import openslide
sys.path.append(os.getcwd())


def write_single_wsi(path, q):

    wsi_path, json_path = path
    t1 = time.time()

    basename = os.path.basename(wsi_path)
    json_data = json.load(open(json_path))
    coords = json_data['coords']

    wsi = openslide.OpenSlide(wsi_path)

    print('reading image {}, {}'.format(wsi_path, coords[0][0]))

    for coord in coords[0]:
        (x, y), level, (patch_size_x, patch_size_y) = coord
        coord = (int(x), int(y)), int(level), (int(patch_size_x), int(patch_size_y))
        patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
            basename=basename,
            x=x,
            y=y,
            level=level,
            patch_size_x=patch_size_x,
            patch_size_y=patch_size_y)
        try:
            patch = wsi.read_region(*coord).convert('RGB')
        except Exception as e:
            print(coord, wsi_path)
            print(e)
            raise ValueError('some thing wrong')

        img_byte_arr = io.BytesIO()
        patch.save(img_byte_arr, format='jpeg')
        img_byte_arr = img_byte_arr.getvalue()
        record = patch_id.encode(), img_byte_arr
        q.put(record)

    print(time.time() - t1, len(coords[0]), wsi_path)

def get_file_path(settings):
    csv_path = settings.file_list_csv
    with open(csv_path, 'r') as csv_file:
        for row in csv.DictReader(csv_file):
            slide_id = row['slide_id']

            name = os.path.splitext(slide_id)[0]
            json_path = os.path.join(settings.json_dir, name + '.json')
            wsi_path = os.path.join(settings.wsi_dir, slide_id)

            path = wsi_path, json_path
            yield path

def get_num_patch(path):
    _, json_path = path
    json_data = json.load(open(json_path, 'r'))
    return len(json_data['coords'][0])

def get_total_num_patches(settings):
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.map(get_num_patch, get_file_path(settings))
    return sum(results)

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
    else:
        raise ValueError('wrong dataset')

    patch_path = settings.patch_dir
    if not os.path.exists(patch_path):
        os.makedirs(patch_path)

    # q = Queue()
    q = Manager().Queue()
    # fn = partial(write_single_wsi, settings=settings, q=q)
    fn = partial(write_single_wsi, q=q)

    num_process = mp.cpu_count()

    print('calculating total number of patches....')
    total_num_patches = get_total_num_patches(settings)
    print('done, total {} num of patcehs'.format(total_num_patches))

    pool = mp.Pool(processes=num_process)
    pool.map_async(fn, get_file_path(settings))


    db_size = 1 << 42
    print(settings.patch_dir)
    env = lmdb.open(settings.patch_dir, map_size=db_size)
    # env = lmdb.open('/data/smb/syh/WSI_cls/default_clam_code/CLAM/tmp_lmdb', map_size=db_size)

    t1 = time.time()

    with env.begin(write=True) as txn:

        count = 0
        while True:
            try:
                record = q.get()
                count += 1
                txn.put(*record)

                if count % 1000 == 0:
                    print('processed {} num of patches, average process time {}'.format
                        (
                            count,
                            (time.time() - t1) / count
                        )
                    )

                if count == total_num_patches:
                    break

            except Exception as e:
                print(e)
                break