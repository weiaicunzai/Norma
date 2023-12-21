import os
import sys
sys.path.append(os.getcwd())
import torch
from torchvision import transforms
from PIL import Image
from model.vit import  get_vit256# 导入 vit_small 函数
from functools import partial
import torch.nn as nn
from dataset.dataloader_simple import  CAMLON16Dataset
import  lmdb
import json
import albumentations as A
import albumentations.pytorch as AP
import torch.distributed as dist


# 1. 实例化模型
def Scale_VIT(data_set,lmdb_dataset_path,batch_size,lmdb_save):
    # model = vit_small()

    # 2. 加载预训练权重
    # pretrained_weights = torch.load('vit256_small_dino.pth')
    # print(pretrained_weights)
    # model.load_state_dict(pretrained_weights)
    model = get_vit256('vit256_small_dino.pth')

    trans = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            AP.transforms.ToTensorV2(),
        ]
    )

    # 准备数据集
    dataset = CAMLON16Dataset(data_set, lmdb_dataset_path, batch_size, transforms=trans, drop_last=False, dist=dist)  # 使用正确的参数初始化你的数据集

    # 创建 LMDB 环境
    env = lmdb.open(lmdb_save, map_size=int(1e12))



    import time
    # 提取特征并保存
    model.eval()
    with torch.no_grad():
        count = 0
        t1 = time.time()
        for data in dataset:
            # 读取图像 patch 和 patch_id
            img, patch_id = data['img'], data['patch_id']

            # print(patch_id)
            # 将图像 patch 转为 tensor 并通过 ViT 模型提取特征
            # img_tensor = torch.from_numpy(img).unsqueeze(0)  # 增加批次维度
            # # ature = model(img_tensor)
            feature = model(img)

            count += img.shape[0]

            print(count / (time.time() - t1))

            # 取出 cls_token 作为特征
            for feat, p_id in zip(feature, patch_id):
                # cls_token = feature[:, 0].squeeze()

                # print()
                #将特征转换为 byte string
                feature_str = json.dumps(feat.cpu().tolist()).encode()

                # 保存到 LMDB
                with env.begin(write=True) as txn:
                  txn.put(p_id.encode(), feature_str)

    env.close()

from conf.camlon16 import settings

Scale_VIT(
    data_set='train',
    lmdb_dataset_path=settings.train_dirs['lmdb'][0],
    batch_size=64,
    lmdb_save='/data/ssd1/by/CAMELYON16/training_feat'
)