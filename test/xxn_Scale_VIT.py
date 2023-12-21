import torch
from torchvision import transforms
from PIL import Image
from model.vit import   vit_small  # 导入 vit_small 函数
from functools import partial
import torch.nn as nn
from dataset.dataloader_simple import  CAMLON16Dataset
import  lmdb
import json
# 1. 实例化模型
def Scale_VIT(patch_size,data_set,lmdb_dataset_path,batch_size,lmdb_save):
    model = vit_small(patch_size)

    # 2. 加载预训练权重
    pretrained_weights = torch.load('vit256_small_dino.pth')
    model.load_state_dict(pretrained_weights)

    # 准备数据集
    dataset = CAMLON16Dataset(data_set, lmdb_dataset_path,batch_size)  # 使用正确的参数初始化你的数据集

    # 创建 LMDB 环境
    env = lmdb.open(lmdb_save, map_size=int(1e12))

    # 提取特征并保存
    model.eval()
    with torch.no_grad():
        for data in dataset:
            # 读取图像 patch 和 patch_id
            img, patch_id = data['img'], data['patch_id']

            # 将图像 patch 转为 tensor 并通过 ViT 模型提取特征
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # 增加批次维度
            feature = model(img_tensor)

            # 取出 cls_token 作为特征
            cls_token = feature[:, 0].squeeze()

            # 将特征转换为 byte string
            feature_str = json.dumps(cls_token.tolist()).encode()

            # 保存到 LMDB
            with env.begin(write=True) as txn:
                txn.put(patch_id.encode(), feature_str)

    env.close()
