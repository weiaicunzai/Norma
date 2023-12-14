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

from .vit import vit_small, vit_base
from .mil_head import AttentionHead, AttentionHeadPara, AttentionHeadAdaptive




class MyNet(nn.Module):
    # def __init__(self, n_classes=2, n_dim=384, interval=100, dis_mem_len=64):
    def __init__(self, n_classes=2, n_dim=384, dis_mem_len=512, alpha=-0.1):
        super().__init__()
        self.n_dim = n_dim
        # self.interval = interval
        self.dis_mem_len = dis_mem_len

        if n_dim == 384:
            self.vit = vit_small(num_classes=0)
        # self.vit = vit_base()
        else:
            self.vit = vit_base(num_classes=0)

        # print(sum([p.numel() for p in self.vit.parameters()]))
        # import sys; sys.exit()
        # self.head = AttentionHead(n_dim=n_dim, interval=interval, dis_mem_len=dis_mem_len)
        # self.head = AttentionHeadPara(n_dim=n_dim, interval=interval, dis_mem_len=dis_mem_len)
        self.head = AttentionHeadAdaptive(n_dim=n_dim, dis_mem_len=dis_mem_len)
        self.fc = nn.Linear(n_dim, n_classes)

    def reset(self):
        self.head.reset()

    def forward(self, x, mem, is_last, hook=None):
        # print(x.device, 'x.shape', device)
        # print(x.shape)
        x = self.vit(x)
        # print(x.shape)
        #print(list(self.fc.weight())[0].data.weight)
        # print(self.vit.cls_token[0, 0, 34], self.vit.cls_token.mean())
        # print(x.shape)
        # print(x.shape)
        x, mem = self.head(x, mem, is_last, hook=hook)
        # print(mem.requires_grad)
        # print(x.shape) [64, 384]
        # x = x[:, 0]
        # print(x.shape)
        logits = self.fc(x)

        # if
        # if is_last is not None:
        #     if is_last.sum() > 0:
        #         print('reset {}'.format(is_last.sum()))
        #         self.reset()

        return logits, mem




# output = net(inputs)
# print(output.shape)