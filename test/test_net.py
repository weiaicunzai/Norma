import os
import sys

sys.path.append(os.getcwd())

from model.my_net import  MyNet
import torch


def test_net():
    # from dataset import utils
    # dataloader = utils.build_dataloader('cam16', 'train', False, batch_size=16, num_workers=4)
    net = MyNet(dis_mem_len=10, n_dim=384, interval=100)

    for i in range(500):
        img = torch.randn(32, 3, 256, 256)
        out = net(img)
        print(i, len(net.head.dis_mem[0]),  len(net.head.cand_mem[0]))
        print(out.shape)

    # for data in dataloader:
        # out = net(data)
    return out



test_net()