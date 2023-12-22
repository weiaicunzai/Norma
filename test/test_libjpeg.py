
path = '/data/hdd1/by/tmp_folder/here.jpg'
import glob
import os

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



import time
import torch


tensor1 = torch.randn(64, 512, 384)
tensor2 = torch.randn(64, 512, 384)


t1 = time.time()
for i in range(1000):
    tensor1 == tensor2

print((time.time() - t1) / 1000)