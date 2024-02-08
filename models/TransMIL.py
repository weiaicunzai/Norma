import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        #self.attn = NystromAttention(
        #    dim = dim,
        #    dim_head = dim//8,
        #    heads = 8,
        #    num_landmarks = dim//2,    # number of landmarks
        #    pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        #    residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        #    dropout=0.1
        #)
        self.attn = Attention(
            dim=dim,
            heads = 8,
            dim_head = dim // 8,
            dropout = 0.1
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
        # net_dim = 384
        # self.pos_layer = PPEG(dim=512)
        self.pos_layer = PPEG(dim=net_dim)
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        # self._fc1 = nn.Sequential(nn.Linear(1024, net_dim), nn.ReLU())
        # input_dim = 384
        input_dim = 1024
        self._fc1 = nn.Sequential(nn.Linear(input_dim, net_dim), nn.ReLU())
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, net_dim))
        self.n_classes = n_classes

        # self.layer1 = TransLayer(dim=net_dim)
        # self.layer2 = TransLayer(dim=net_dim)

        self.layer1 = Transformer(
            dim= net_dim,
            depth=1,
            heads= 8,
            dim_head= net_dim // 8,
            mlp_dim= net_dim // 8 * 4,
            dropout = 0.1
        )
        self.layer2 = Transformer(
            dim=net_dim,
            depth=1,
            heads= 8,
            dim_head= net_dim // 8,
            mlp_dim= net_dim // 8 * 4,
            dropout = 0.1
        )

        self.layer3 = Transformer(
            dim= net_dim,
            depth=1,
            heads= 8,
            dim_head= net_dim // 8,
            mlp_dim= net_dim // 8 * 4,
            dropout = 0.1
        )

        # self.norm = nn.LayerNorm(net_dim)

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
        # print(h.shape, 'ccccccccccc')
        # torch.Size([2, 1024, 384])

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


        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]


        h = self.layer3(h) #[B, N, 512]


        mems = self._update_mems(mems, h[:, 0].unsqueeze(dim=1))

        #---->cls_token
        # h = self.norm(h)[:,0]
        h = h[:,0]

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
