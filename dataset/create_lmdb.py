import lmdb
import os
import glob
import sys
import json
import io
sys.path.append(os.getcwd())
from PIL import Image

import openslide

from conf import camlon16



# def tif2json(tif_path):
    # tif_path.replace('training', 'training_json').replace('.')
    # if 'tumor' in


# def write


def create_lmdb(data_set, lmdb_path):
    #wsi_path = os.path.join()

    # for i in glob.iglob(os.path.join(wsi_path, "**", '*.tif'), recursive=True):
        # print(i)



# def camlon16_wsis(data_set, direction=-1):
    if data_set == 'train':
        dirs = camlon16.train_dirs
    else:
        dirs = camlon16.test_dirs


    db_size = 1 << 40
    env = lmdb.open(lmdb_path, map_size=db_size)


    # wsis = []
    # txn = lmdb.open(lmdb_path)

    count = 0
    for wsi_dir, json_dir in zip(dirs['wsis'], dirs['jsons']):
        for wsi_path in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
            # print(wsi_path)
            basename = os.path.basename(wsi_path)
            json_path = os.path.join(json_dir, basename.replace('.tif', '.json'))
            print(wsi_path, json_path)

            # with env.begin(write=True) as txn:
                # txn.put()

            json_data = json.load(open(json_path))
            # filename = json_data['filename']
            # txn.put(json_data.encode(), img_buff)
            # print(filename)
            # print(json_data.keys())
            coords = json_data['coords']
            # print(len(coords))

            wsi = openslide.OpenSlide(wsi_path)

            print('saving image {}'.format(wsi_path))

            with env.begin(write=True) as txn:
                for coord in coords[0]:
                    count += 1
                    # print(coord)
                    (x, y), level, (patch_size_x, patch_size_y) = coord
                    # print(x, y, level, patch_size_x, patch_size_y)
                    patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
                        basename=basename,
                        x=x,
                        y=y,
                        level=level,
                        patch_size_x=patch_size_x,
                        patch_size_y=patch_size_y)
                    # print(basename)
                    # print(patch_id)
                    patch = wsi.read_region(*coord).convert('RGB')
                    # print(patch.size)
                    # patch.save('tmp/{}.jpg'.format(count))
                    img_byte_arr = io.BytesIO()
                    patch.save(img_byte_arr, format='jpeg')
                    img_byte_arr = img_byte_arr.getvalue()
                    # print(type(img_byte_arr))
                    # ss = Image.open(io.BytesIO(img_byte_arr))
                    # print(ss)
                    # ss.save('tmp')
                    # ss.save('tmp/{}_after.jpg'.format(count))
                    # print(patch_id.encode())
                    # print(patch_id, count)
                    txn.put(patch_id.encode(), img_byte_arr)





            # import sys; sys.exit()



            # import sys; sys.exit()
            # print(set(coords[3]) - set(coords[6]))
            # print(coor)
            # for i in coords[0]:
                # print(i)

            # for c in coords[0]:
            #     print(coords[3].index(c))

            # print(len(coords[0]))
            # import sys; sys.exit()








# wsi_path = '/data/ssd1/by/CAMELYON16/training/'
# json_path = '/data/ssd1/by/CAMELYON16/training_json/'
# lmdb_path = '/data/ssd1/by/CAMELYON16/training_lmdb'
lmdb_path = '/data/ssd1/by/CAMELYON16/testing_lmdb'


# create_lmdb(wsi_path, json_path, lmdb_path)
# create_lmdb('train', lmdb_path)
create_lmdb('test', lmdb_path)