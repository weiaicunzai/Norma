import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        # net_dim = 512
        net_dim = 512
        # self.pos_layer = PPEG(dim=512)
        self.pos_layer = PPEG(dim=net_dim)
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self._fc1 = nn.Sequential(nn.Linear(1024, net_dim), nn.ReLU())
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, net_dim))
        self.n_classes = n_classes
        # self.layer1 = TransLayer(dim=512)
        self.layer1 = TransLayer(dim=net_dim)
        # self.layer2 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=net_dim)
        # self.norm = nn.LayerNorm(512)
        self.norm = nn.LayerNorm(net_dim)
        # self._fc2 = nn.Linear(512, self.n_classes)
        self._fc2 = nn.Linear(net_dim, self.n_classes)


        # self.mem_length = 20
        # self.mem_length = int(512 * 78 * 2 / (512 * 2) / 2)
        # self.mem_length = 4
        # self.mem_length = 2
        # self.mem_length = 8
        # self.mem_length = 8
        # self.mem_length = 78
        # self.mem_length = int(512 * 78 * 2 / (512 * 2) / 2)
        self.mem_length = 39
        print('net_dim', net_dim, 'mem_length', self.mem_length)

    def _update_mems(self, mems, h):

        if mems is None:
            # print('mem updated before', mems)
            mems = h.detach().clone()
        else:
            # print('mem updated before', mems[0, :, 55])
            mems = torch.cat([mems, h], dim=1)
            mems = mems[:, -self.mem_length:, :].detach().clone()

        # print('mem shape', mems.shape)
        # print('mem updated after', mems[0, :, 55])
        return mems

    def forward(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]

        # is_last = kwargs['is_last'].float() #[B, n, 1024]
        mems = kwargs.get('mems', None)

        h = self._fc1(h) #[B, n, 512]

        if mems is not None:
            h = torch.cat((h, mems), dim=1)

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        mems = self._update_mems(mems, h[:, 0].unsqueeze(dim=1))

        #---->cls_token
        h = self.norm(h)[:,0]

        # print(mems.shape)

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'mems': mems}
        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
