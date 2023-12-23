import sys
# sys.path.append('/tmp/pycharm_project_945')
import os
sys.path.append(os.getcwd())
from conf.camlon16 import settings
from dataset.dataloader_simple import  CAMLON16DatasetFeat
import torch.distributed as dist
import torch

dataset=CAMLON16DatasetFeat( data_set='val',
    # lmdb_path='/data/ssd1/xuxinan/CAMELYON16/testing_feat',
    # lmdb_path='/data/ssd1/by/CAMELYON16/testing_lmdb/',
    lmdb_path='/data/ssd1/by/CAMELYON16/testing_feat',
    batch_size=128,
    seq_len=256,
    dist=dist)
count=0
import time
t1 = time.time()

import cProfile, pstats
from pstats import SortKey


import cv2

sortby = SortKey.CUMULATIVE

dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None, persistent_workers=True)


# with cProfile.Profile() as pr:
from viztracer import VizTracer
with VizTracer(output_file="optional.json") as tracer:

  for data in dataset:
# = -
# for iter, data in enumerate(dataloader):

    # print( data['img'].shape)
    # print(data.keys())
    # count=count + data['img'].shape[0]
    count += 1
    # print(data['img'].shape, '11111')
    # print((t1 - time.time())  / (iter + 1e-8))
    if count > 30:
        break

    # ps = pstats.Stats(pr).sort_stats(sortby)
    # ps.print_stats()
