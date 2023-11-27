import torch
import torch.nn as nn


# class AA:
#     def __init__(self):
#         self.aa = []

#     def init(self, cc):
#         for i in range(10):
#             cc.append(i)

#     def update(self):
#         self.init(self.aa)


# aa = AA()

# print(aa.update())
# print(aa.aa)
# test = [
#     {'min': 10, 'total': 20},
#     {'min': 14, 'total': 20},
#     {'min': 13, 'total': 20},
# ]


# for i in test:
#     i['min'] += 100


# print(test)

from .vit import vit_small
from .mil_head import AttentionHead




class MyNet(nn.Module):
    def __init__(self, n_classes=2, n_dim=384, interval=100, dis_mem_len=64):
        super().__init__()
        self.n_dim = n_dim
        self.interval = interval
        self.dis_mem_len = dis_mem_len
        self.vit = vit_small()
        self.head = AttentionHead(n_dim=n_dim, interval=interval, dis_mem_len=dis_mem_len)
        self.fc = nn.Linear(n_dim, n_classes)

    def forward(self, x, is_last=None):
        # print(x.device, 'x.shape', device)
        x = self.vit(x)
        # x = self.head(x)
        logits = self.fc(x)

        # if
        # if is_last.sum() > 0:
            # print('reset {}'.format(is_last.sum()))
            # self.head.reset()

        return logits
