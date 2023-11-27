
from PIL import Image

import numpy as np
import cv2


# img_path = '/data/hdd1/by/tmp_folder/test.jpg'


# img_cv2 = cv2.imread(img_path)
# img_cv2_hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)

# print(img_cv2_hsv[:, :, 2].max())
# # print(img_cv2_hsv[:, :, 0].max())


# img_pil = Image.open(img_path)
# img_pil = img_pil.convert('HSV')

# # H, S, V = img_pil.split()
# img_pil = np.array(img_pil)
# # print(img_pil[:, :, 0])
# print(img_pil[:, :, 0].max())

import torchmetrics
import torch

acc = torchmetrics.Accuracy(task="binary", num_classes=2)
# target = torch.tensor([0, 1, 2])
# preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
target = torch.tensor([0, 1, 0, 1, 0, 1])
preds = torch.tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])


class_wise_acc = torchmetrics.wrappers.ClasswiseWrapper(torchmetrics.Accuracy(task='multiclass', num_classes=2, average=None))

# c = torchmetrics.wrappers.ClasswiseWrapper(torchmetrics.Accuracy(task='multiclass', num_classes=3, average=None), labels=["11", "22", "33"], prefix='cc')
c = torchmetrics.Accuracy(task='multiclass', num_classes=3, average='micro')

# preds = torch.randn(10, 3).softmax(dim=-1)
# target = torch.randint(3, (10,))
target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
preds = torch.tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
                [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])


target = torch.tensor([0, 1, 0, 1, 0, 1])
preds = torch.tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])

print(target.shape, preds.shape)
print(class_wise_acc(preds, target))

target = torch.tensor([1, 1, 2])
preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])

print(c(preds, target))