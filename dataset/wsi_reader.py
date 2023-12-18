
import os
import sys
import glob

# import cv2
import lmdb

from .wsi import WSILMDB, WSIJSON
from conf import camlon16



# def camlon16_wsi_filenames(data_set):
#     outputs = []
#     if data_set == 'train':
#         dirs = _train_dirs
#     else:
#         dirs = _test_dirs


#     for wsi_dir, json_dir in zip(dirs['wsis'], dirs['jsons']):
#         for wsi_path in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
#             basename = os.path.basename(wsi_path)
#             json_path = os.path.join(json_dir, basename.replace('.tif', '.json'))
#             outputs.append({
#                 'wsi': wsi_path,
#                 'json': json_path
#             })

#     return outputs


# def camlon16_wsis(wsi_filenames, label_fn, direction):
class CAMLON16MixIn:

    def camlon16_wsis(self, data_set, direction=-1):
        # print(direction)
        if data_set == 'train':
            dirs = camlon16.train_dirs
            # patch_dir = camlon16.train_dirs
        else:
            dirs = camlon16.test_dirs



        wsis = []
        # for wsi_dir, json_dir in zip(dirs['wsis'], dirs['jsons']):
        # lmdb_path = dirs['lmdb'][0]

        # env = lmdb.open(db_path, readonly=True)

        count = 0
        # env = lmdb.open(lmdb_path, readonly=True, lock=False)
        for json_dir in dirs['jsons']:
            # for wsi_path in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
                # print(wsi_path)
                # basename = os.path.basename(wsi_path)


            for json_path in glob.iglob(os.path.join(json_dir, '**', '*.json'), recursive=True):
                # json_path = os.path.join(json_dir, basename.replace('.tif', '.json'))
                # outputs.append({
                #     'wsi': wsi_path,
                #     'json': json_path
                # })
                count += 1

                # import time
                # t1 =time.time()

                # this file do not have
                if 'test_114' in json_path:
                    continue

                print(json_path, count)
                # wsi = WSILMDB(
                wsi = WSIJSON(
                    # filename['wsi'],
                    # wsi_path,
                    # mask_path(wsi_path),
                    json_path,
                    # env=env,
                    # patch_size=512,
                    # at_mag=5,
                    # random_rotate=True,
                    # label_fn=label_fn,
                    direction=direction,
                    patch_json_dir=dirs['patch_level'][0],
                )
                # t2 = time.time()
                # print('total: ', t2 - t1)


                # wsis.append(wsi)

                if wsi.num_patches > 0:
                    wsis.append(wsi)
                else:
                    print(json_path, 'is 0')

        return wsis


    # def mask_path(wsi_path):
    #     mask_path = wsi_path.replace('training', 'training_mask')
    #     mask_path = mask_path.replace('.tif', '.png')
    #     return mask_path

    # wsis = []
    # #for i in glob.iglob(os.path.join())
    # for filename in wsi_filenames:

    #           # self.wsi_filenames.append(i)
    #           # print(wsi_path, label_fn(wsi_path))
    #           # wsis.append(
    #         wsi = WSI(
    #             filename['wsi'],
    #             # mask_path(wsi_path),
    #             filename['json'],
    #             # patch_size=512,
    #             # at_mag=5,
    #             # random_rotate=True,
    #             # label_fn=label_fn,
    #             direction=direction
    #         )

    #         # print(filename)
    #         if wsi.num_patches > 0:
    #             wsis.append(wsi)

    #         else:
    #             print(wsi_path, 'is 0')

    #         # outputs.append(wsi)

    # # return outputs
    # return wsis


def camlon16_wsis(data_set, direction=-1):
        # print(direction)
        if data_set == 'train':
            dirs = camlon16.train_dirs
            # patch_dir = camlon16.train_dirs
        else:
            dirs = camlon16.test_dirs



        wsis = []
        # for wsi_dir, json_dir in zip(dirs['wsis'], dirs['jsons']):
        lmdb_path = dirs['lmdb'][0]

        # env = lmdb.open(db_path, readonly=True)

        count = 0
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        for json_dir in dirs['jsons']:
            # for wsi_path in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
                # print(wsi_path)
                # basename = os.path.basename(wsi_path)


            for json_path in glob.iglob(os.path.join(json_dir, '**', '*.json'), recursive=True):
                # json_path = os.path.join(json_dir, basename.replace('.tif', '.json'))
                # outputs.append({
                #     'wsi': wsi_path,
                #     'json': json_path
                # })
                count += 1

                # import time
                # t1 =time.time()

                # this file do not have
                if 'test_114' in json_path:
                    continue

                print(json_path, count)
                wsi = WSILMDB(
                # wsi = WSIJSON(
                    # filename['wsi'],
                    # wsi_path,
                    # mask_path(wsi_path),
                    json_path,
                    env=env,
                    # patch_size=512,
                    # at_mag=5,
                    # random_rotate=True,
                    # label_fn=label_fn,
                    direction=direction,
                    patch_json_dir=dirs['patch_level'][0],
                )
                # t2 = time.time()
                # print('total: ', t2 - t1)


                # wsis.append(wsi)

                if wsi.num_patches > 0:
                    wsis.append(wsi)
                else:
                    print(json_path, 'is 0')

        return wsis
