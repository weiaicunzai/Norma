import sys
# sys.path.append('/tmp/pycharm_project_945')
import os
sys.path.append(os.getcwd())
from conf.camlon16 import settings
from dataset.dataloader_simple import  CAMLON16DatasetFeat
import torch.distributed as dist
import torch

dataset=CAMLON16DatasetFeat(
    # data_set='val',
    data_set='train',
    # lmdb_path='/data/ssd1/xuxinan/CAMELYON16/testing_feat',
    # lmdb_path='/data/ssd1/by/CAMELYON16/testing_lmdb/',
    # lmdb_path='/data/ssd1/by/CAMELYON16/testing_feat',
    # lmdb_path='/data/ssd1/by/CAMELYON16/testing_feat1/',
    lmdb_path='/data/ssd1/by/CAMELYON16/training_feat1/',
    batch_size=32,
    seq_len=256,
    dist=dist,
    drop_last=False,
    all=True)
count=0
import time

import cProfile, pstats
from pstats import SortKey


import cv2

sortby = SortKey.CUMULATIVE

dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None, persistent_workers=True)


# with cProfile.Profile() as pr:
# from viztracer import VizTracer
# with VizTracer(output_file="optional1.json") as tracer:

t1 = time.time()
import lmdb
env = lmdb.open('/data/ssd1/by/CAMELYON16/training_feat1/', readonly=True)
print(env.stat())
# for i in range(10):
# for data in dataset:
# = -
import cProfile, pstats
from pstats import SortKey


import cv2

sortby = SortKey.CUMULATIVE

with cProfile.Profile() as pr:
# from viztracer import VizTracer
# with VizTracer(output_file="optional1.json") as tracer:
for e in range(3):
 print(e, e, e, e, e, e, e)
 for iter, (data, lens) in enumerate(dataloader):

    # print( data['img'].shape)
    # print(data.keys())
    # count=count + data['img'].shape[0]
    count += 1
    # print(data['img'].shape, data['is_last'])
    # print(len(dataloader.dataset.cache.keys()))
    # print(lens)
    # print(data['img'].shape, '11111')
    print((time.time() - t1)  / (count + 1e-8))
    # if count > 15:
        # break

    # ps = pstats.Stats(pr).sort_stats(sortby)
    # ps.print_stats()
