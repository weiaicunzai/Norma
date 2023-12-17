import os
import sys

sys.path.append(os.getcwd())

from model.my_net import  MyNet
import torch


def _update_mem(wsi_feat, attn_score):
        with torch.no_grad():
            len = wsi_feat.shape[1]
            if len > 3:
                # attn_idx = torch.topk(attn_score, k=wsi_feat.shape[1] - 1, dim=1, sorted=False)[1]
                # new_attn_idx = attn_idx.expand(attn_idx.shape[0], attn_idx.shape[1], wsi_feat.shape[-1])
                # new_mem = torch.gather(wsi_feat, dim=1, index=new_attn_idx)
                attn_idx = torch.min(attn_score, dim=1)[1]
                mask = torch.ones(wsi_feat.shape[:2]).scatter_(1, attn_idx, 0.)
                new_mem = wsi_feat[mask.bool()].view(wsi_feat.shape[0], -1, wsi_feat.shape[-1])




                return new_mem
            else:
                return wsi_feat.detach()


def test_update_mem():

    # for i in range(100):
        wsi_feat = torch.randn(2, 4, 3)
        attn_score = torch.randn(2, 4, 1).softmax(dim=1)
        print(attn_score)
        print(wsi_feat)
        # attn_index = torch.topk(attn_score, k = wsi)

        # for index in
        mem = _update_mem(wsi_feat, attn_score)
        print(mem)


def test_net():
    # from dataset import utils
    # dataloader = utils.build_dataloader('cam16', 'train', False, batch_size=16, num_workers=4)
    # net = MyNet(dis_mem_len=10, n_dim=384, interval=100).cuda()
    # net = MyNet(dis_mem_len=512, n_dim=384).cuda()
    net = MyNet(dis_mem_len=256, n_dim=384).cuda()
    is_last = torch.zeros(4).cuda()

    # mem = None
    # mems =

# net = MyNet().cuda()
    mems = {
        'feat': None,
        'freq': None,
        'min': None
    }

    with torch.no_grad():
     for i in range(500):
        img = torch.randn(4, 3, 256, 256).cuda()
        out, mems = net(img, mems, is_last)
        # print(i, len(net.head.dis_mem[0]),  len(net.head.cand_mem[0]))
        # print(out.shape)
        # print()
        for k, v in mems.items():
            if v is not None:
                print(k, v.shape)


    # for data in dataloader:
        # out = net(data)
    return out



test_net()
# test_update_mem()