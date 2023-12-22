import sys
sys.path.append('/tmp/pycharm_project_945')
from conf.camlon16 import settings
from dataset.dataloader_simple import  CAMLON16DatasetFeat
import torch.distributed as dist
dataset=CAMLON16DatasetFeat( data_set='val',
    lmdb_path='/data/ssd1/xuxinan/CAMELYON16/testing_feat',
    batch_size=64,
    seq_len=16,
    dist=dist)
count=0
for data in dataset:

    print( data['img'])
    count=count+1