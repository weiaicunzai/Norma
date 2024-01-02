import argparse
import csv
import io
import multiprocessing as mp
import json
import os
import time
import sys
from functools import partial
from multiprocessing import Process, Queue


import lmdb
import openslide
sys.path.append(os.getcwd())


def write_single_wsi(path, settings, q):

    wsi_path, json_path = path
    t1 = time.time()

    basename = os.path.basename(wsi_path)
    json_data = json.load(open(json_path))
    coords = json_data['coords']

    wsi = openslide.OpenSlide(wsi_path)

    print('reading image {}'.format(wsi_path))

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
        q.put((patch_id.encode(), img_byte_arr))


    print(time.time() - t1, len(coords[0]))

def get_file_path(settings):
    csv_path = settings.file_list_csv
    with open(csv_path, 'r') as csv_file:
        for row in csv.DictReader(csv_file):
            slide_id = row['slide_id']
            name = os.path.splitext(slide_id)[0]
            json_path = os.path.join(settings.json_dir, name + '.json')
            wsi_path = os.path.join(settings.wsi_dir, slide_id)

            yield wsi_path, json_path

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()



if __name__ == '__main__':
    args = get_args_parser()
    if args.dataset == 'brac':
        from conf.brac import settings

    patch_path = settings.patch_dir
    if not os.path.exists(patch_path):
        os.makedirs(patch_path)

    q = Queue()
    fn = partial(write_single_wsi, settings=settings, q=q)

    db_size = 1 << 40
    env = lmdb.open(settings.patch_dir, map_size=db_size)
    count = 0
    t1 = time.time()
    num_process = 16
    with env.begin(write=True) as txn:

        proc = []
        for path in get_file_path(settings):
            proc.append(
                Process(target=fn, args=(path,))
            )

            if len(proc) == num_process:
                for p in proc:
                    p.start()

                time.sleep(20)
                while True:
                    try:
                        # if after 10 seconds, still no data
                        # then means the process ends
                        record = q.get(timeout=10)
                        txn.put(*record)
                        count += 1
                        if count % 1000 == 0:
                            print('time', (time.time() - t1) / count, 'total', count)
                    except:
                        print('end of reading {} processes'.format(len(proc)))
                        break


                # wait untill process ends
                for p in proc:
                    p.join()

                # clear prc
                proc = []

        # if last p less len(proc)
        for p in proc:
            p.start()

        time.sleep(10)
        while True:
            try:
                # if after 10 seconds, still no data
                # then means the process ends
                record = q.get(timeout=10)
                txn.put(*record)
                if count % 1000 == 0:
                    print('time', (time.time() - t1) / count, 'total', count)
            except:
                print('end of reading {} processes'.format(len(proc)))
                break

        for p in proc:
            p.join()
