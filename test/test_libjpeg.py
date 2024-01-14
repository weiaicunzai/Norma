
path = '/data/hdd1/by/tmp_folder/here.jpg'
import glob
import os
import sys
sys.path.append(os.getcwd())

from PIL import Image


# a = Image.open(path).convert('RGB')
# jpg_path = '/data/ssd1/xuxinan/flower102jpg'
# save_path = '/data/ssd1/by/folowler'


# for img_path in glob.iglob(os.path.join(jpg_path, '**', '*.jpg'), recursive=True):
#     # print(img)
#     img = Image.open(img_path)

#     basename = os.path.basename(img_path).replace('jpg', 'png')

#     img.save(os.path.join(save_path, basename))
    # print()


# class PKGInfo:
    # def __init__(self, file_path):
        # self.files = []

    # def test(self):
# import collections

# def parse_file(path):
#     res = {}
#     # res[pkg] =
#     # PKG = collections.namedtuple('PKG', ['name', 'version', 'build', 'channel'])
#     with open(path) as f:
#         for line in f.readlines():
#             pkg = line.split()
#             # print(pkg)
#             res[pkg[0]] = line.strip()
#             # print(res[pkg[0]])

#     return res

# myenv = parse_file('conda_myenv')
# # print(myenv)
# test1 = parse_file('conda_test1')
# torch113 = parse_file('conda_torch1.13')
# torchlts = parse_file('conda_torchlts')
# gzm = parse_file('conda_gzm')
# memvit = parse_file('conda_memvit')
# py37 = parse_file('conda_py37')
# pyszy = parse_file('conda_pyszy')

# slow = [
#     test1, torch113, memvit
# ]

# fast = [torchlts, myenv, gzm, py37, pyszy]

# all_fast_have = []
# all_slow_have = []

# pkg_names = []
# with open('sss.txt') as f:
#     for line in f.readlines():
#         # print(line)
#         if 'The following' in line:
#             # print(line)
#             continue
#         if line.strip():
#             pkg = line.split()[0].replace('~', '')
#             # if 'llvm-openmp' == pkg:
#                 # print(pkg)
#             pkg_names.append(pkg.strip())

# def in_all(pkg_name, value, all_envs):
#     if_in = True
#     for env in all_envs:
#         # line
#         line = env.get(pkg_name)
#         if pkg_name == 'llvm-openmp':
#             print('ccccc', pkg_name)
#         if line != value:
#             if_in = False
#         # if pkg_name not in env:
#             # print(env)
#             # if_in = False

#     return if_in


# count = 0
# # for env in fast:
# #     for key, value in myenv.items():
# #         count += 1
# #         # print(key)
# #         in_slow = in_all(key, value, slow)
# #         in_fast = in_all(key, value, fast)
# #         print(key, 'in fast', in_fast, 'in slow', in_slow)

# count = 0
# for pkg in pkg_names:
#     # print(pkg)
#         # import sys; sys.exit()
#     # print(key, value)
#     slow_pkgs = []
#     fast_pkgs = []

#     # print('fast:')
#     for env in fast:
#         # print(pkg, env.get(pkg, 'None'))
#         fast_pkgs.append(env.get(pkg, 'None'))
#     # print('slow:')
#     for env in slow:
#         # print(pkg, env.get(pkg, 'None'))
#         # print(pkg, env.get(pkg, 'None'))
#         slow_pkgs.append(env.get(pkg, 'None'))

#     # print()


#     show = True
#     for s_pkg in slow_pkgs:
#         if s_pkg in fast_pkgs:
#             show = False

#     # if f_pkg in fast_pkgs:
#         # if fast_pkgs is 'None':

#     # if 0 < fast_pkgs.count('None') < len(fast_pkgs):
#         # show = False

#     # if pkg == 'llvm-openmp':
#     #     show = True
#     if show:
#         count += 1
#         print('slow')
#         for pkg in slow_pkgs:
#             print(pkg)

#         print('fast')
#         for pkg in fast_pkgs:
#             print(pkg)


#         # for pkg in fast:
#         #     print(pkg)

#     print('-----------------')

# print(count)


path = '/data/ssd1/by/CAMELYON16/training_json/tumor/patch_size_512_at_mag_20_patch_label/'
path = '/data/ssd1/by/CAMELYON16/testing/jsons/patch_size_512_at_mag_20_patch_label/'

import json
import os
import glob

# for i in glob.iglob(os.path.join(path, '*.json')):
#     # print(i)
#     count = 0
#     tumor = 0
#     data = json.load(open(i, 'r'))
#     #print(data)
#     # print(data.keys())
#     # counnnn = 0
#     for idx, (k, v) in enumerate(data.items()):
#         # print(k, v)
#         if v == 1:
#             tumor += 1

#         if idx % 256  == 0:
#             print(tumor)


#         count += 1

#     # print(tumor, count / 512, count)
#     print(count / 1024, tumor, count)


# def aaa(max_len, seq_len):
#     reminder = max_len % seq_len
#     # assert max_len > 0
#     if reminder != 0:
#     # reminder = max_len
#         max_len += seq_len - reminder
#     return max_len


# print(aaa(0, 32))



# import time
# import torch
# import random
# import struct


# # tensor1 = torch.randn(64, 512, 384)
# # tensor2 = torch.randn(64, 512, 384)

# test1 = [random.random() for _ in range(384)]
# test2 = [random.random() for _ in range(384)]

# times = 1000

# # for i in test1:
# d = struct.pack('384f', *test1)

# t1 = time.time()
# for i in range(times):
#     # tensor1 == tensor2
#     # struct.pack('f', )
#     # print(d )
#     c = struct.unpack_from('384f', d)
#     # print(c)

# c = (time.time() - t1) / times
# print(c)
# import json
# str_test1 = json.dumps(test1)

# t1 = time.time()
# for i in range(times):
#     json.loads(str_test1)


# d = (time.time() - t1) / times
# print(d)

# print(d / c)

# import lmdb
# import struct
# lmdb_path = '/data/ssd1/by/CAMELYON16/testing_feat'

# env = lmdb.open(lmdb_path, readonly=True)
# with env.begin(write=False) as txn:
#     # with txn.cursor() as curs:
#         # print()'key is:', curs.get('key'))

#     # length = txn.stat()['entries']
#     #keys = list(txn.cursor().iternext(values=False))
#     #for key in keys:
#     #    value = txn.get(key)
#     #    # print(len(value))
#     #    d = struct.unpack('384f', value)
#     #    # print(d)
#     #    print(len(d))
#     # test_036.tif_60416_83968_1_512_512
#     print(env.path())
#     stream = txn.get('test_036.tif_60416_83968_1_512_512'.encode())
#     print(len(stream))
#     d = struct.unpack('384f', stream)
#     print(len(d))



    #with txn.cursor() as curs:
    #    # do stuff
    #    print('key is:', curs.get('key'))

from dataset.wsi import WSIJSON
from conf.camlon16 import settings
import csv
import pandas


csv_file = settings.file_list_csv
# for wsi_path, json_path in os.path.join()
# from datase
df = pandas.read_csv(settings.file_list_csv)

# print(df['slide_id'])
seq_len = 512 * 2
num_seq = 0
num_contains_tumor_seq = 0
num_right_label = 0
num_wrong_label = 0


print('cccccccccccccccccccccccccccc')

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

for slide_id in df['slide_id']:
    # print(row)
    wsi_path = os.path.join(settings.wsi_dir, slide_id)
    json_path = os.path.join(settings.json_dir, os.path.splitext(slide_id)[0] + '.json')

    if 'test_114' in slide_id:
        continue

    wsi = WSIJSON(
        json_path=json_path,
        direction=1,
        # patch_json_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/'
        # patch_json_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/patch_jsons/patch_size_512_at_mag_20_patch_label/'
        patch_json_dir=settings.patch_label_dir,
    )

    # num_patches = 0
    seq_label = 0
    seq = []
    # print(wsi.num_patches)
    count = 0
    for i in cycle(wsi):
        count += 1
        # num_patches += 1
        # print(i)
        # if i['label'] == 0:
        wsi_label = int(i['label'])
        if wsi_label != 1:
            break
        # if i['label'] == 0 :
            # num_right_label +=

        # print(type(wsi_label))
        # if 'tumor' in slide_id:
            # print(i.get('p_label'))

        patch_label = int(i.get('p_label', 0))
        if patch_label == 1:
            # print(patch_label)
            print(slide_id)
            print(

                'patch_label is ', patch_label
            )
        seq.append(patch_label)
        # print(patch_label)
        # if wsi_label == 1:
            # print(seq)

        # if patch_label == 1:
        #     print(seq)

        if seq_len == len(seq):
            # num_patches
            # sum()
            if wsi_label == 0:
                num_right_label += 1

            else:
                if sum(seq) == 0:
                    num_wrong_label += 1

                else:
                    num_right_label += 1

            num_seq += 1

            seq = []

        if count == 512 * 78:
            break
    print(num_seq, num_right_label, num_wrong_label)

# print('ccccc')
# print(num_seq, num_right_label, num_wrong_label)

# 1024 : 0.39949748743718594
# 2048 : 0.39949748743718594
# 4068: 0.39949748743718594
