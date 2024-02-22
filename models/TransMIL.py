import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        return pos_emb

        # if bsz is not None:
        #     return pos_emb[:,None,:].expand(-1, bsz, -1)
        # else:
        #     return pos_emb[:,None,:]

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

class AttentionMem(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., use_pos=False):
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

        self.use_pos = use_pos

        if self.use_pos:
            self.peg1d_k = PEG1D(dim=dim)
            self.peg1d_v = PEG1D(dim=dim)


    def forward(self, x, mem=None):


        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        if mem is not None:
            # print(mem.shape, k.shape, '1111111111111')
            k = torch.cat((k, mem), dim=1)
            v = torch.cat((v, mem), dim=1)

            # print(self.use_pos)
            if self.use_pos:
                cls_token = k[:, 0, :].unsqueeze(dim=1)
                k = k[:, 1:, :]
                k = self.peg1d_k(k)
                k = torch.cat((cls_token, k), dim=1)

                cls_token = v[:, 0, :].unsqueeze(dim=1)
                v = v[:, 1:, :]
                v = self.peg1d_v(v)
                v = torch.cat((cls_token, v), dim=1)


        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')


        return self.to_out(out)

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


    def forward(self, x, mem=None, pos_emb=None):


        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Attention1(nn.Module):
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


    def forward(self, x, mem=None, pos_emb=None):

        # if mem is not None:
        #     h = torch.cat((x, mem), dim=1)
        #     x = self.norm(h)
        #     # print('q before', q.shape, mem.shape)
        # else:
        #     x = self.norm(x)

        # print(x.shape, pos_emb.shape)
        # x = x + pos_emb

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        if mem is not None:
            k = torch.cat((k, mem), dim=1) + pos_emb
            v = torch.cat((v, mem), dim=1) + pos_emb

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))



        # print(q.shape, 1)
        # print(k.shape, 2)
        # print(v.shape, 3)


        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # if mem is not None:
        #     return self.to_out(out) + h
        # else:
        #     return self.to_out(out)
        # print(out.shape)
        return self.to_out(out)


class Transformer1(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))


    def forward(self, x, mems=None, pos_emb=None):
        hids = []
        for (attn, ff), mem in zip(self.layers, mems):
        # for attn, ff in self.layers:
            # if mem is not None:
                # x = torch.cat((x, mem), dim=1)

            # x = pos_emb + x
            x = attn(x, mem, pos_emb) + x
            # x = attn(x) + x
            # x = attn(x, mem, pos_emb)
            x = ff(x) + x

            # hids.append(x[:,0])

        # return self.norm(x), hids
        return self.norm(x)

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


    def forward(self, x, mems=None, pos_emb=None):
        hids = []
        # for (attn, ff), mem in zip(self.layers, mems):
        for attn, ff in self.layers:
            # if mem is not None:
                # x = torch.cat((x, mem), dim=1)

            # x = pos_emb + x
            # x = attn(x, mem, pos_emb) + x
            x = attn(x) + x
            # x = attn(x, mem, pos_emb)
            x = ff(x) + x

            hids.append(x[:,0])

        # return self.norm(x), hids
        return self.norm(x)


class SelfAttnMEMLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, use_pos=False):
        super().__init__()
        self.norm = norm_layer(dim)
        # self.attn = NystromAttention(
        #    dim = dim,
        #    dim_head = dim//8,
        #    heads = 8,
        #    num_landmarks = dim//2,    # number of landmarks
        #    pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        #    residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        #    dropout=0.1
        # )
        # self.attn = Attention(
        #     dim=dim,
        #     heads = 8,
        #     dim_head = dim // 8,
        #     dropout = 0.1
        # )
        self.attn = AttentionMem(
            dim=dim,
            heads=8,
            dim_head = dim // 8,
            dropout = 0.1,
            use_pos=use_pos
        )

    def forward(self, x, mem=None):
        x = x + self.attn(self.norm(x), mem=mem)

        return x



class TransLayerSelfAttn(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        # self.attn = NystromAttention(
        #    dim = dim,
        #    dim_head = dim//8,
        #    heads = 8,
        #    num_landmarks = dim//2,    # number of landmarks
        #    pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        #    residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        #    dropout=0.1
        # )
        self.attn = Attention(
            dim=dim,
            heads=8,
            dim_head=dim // 8,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

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
        # self.attn = Attention(
        #     dim=dim,
        #     heads = 8,
        #     dim_head = dim // 8,
        #     dropout = 0.1
        # )

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

class PEG1D(nn.Module):
    def __init__(self, dim=512):
        super(PEG1D, self).__init__()
        # self.proj = nn.Conv1d(dim, dim, 7, 1, 7//2, groups=dim)
        # self.proj1 = nn.Conv1d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj1 = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            groups=dim
        )

        # self.proj2 = nn.Conv1d(
        #     in_channels=dim,
        #     out_channels=dim,
        #     kernel_size=3,
        #     stride=1,
        #     padding=(3 - 1) // 2,
        #     groups=dim
        # )
        self.chunk_length = 32

    def pos_emb(self, x):
        B, N, C = x.shape

        chunk_length = self.chunk_length
        num_seq = math.ceil(N / chunk_length)
        # print(num_seq, num_seq * chunk_length, N)

        padded_x = F.pad(x, (0, 0, 0,  num_seq * chunk_length - N))
        padded_x = padded_x.view(B * num_seq, chunk_length, C).permute(0, 2, 1)

        # print(padded_x.shape, 111)
        pos_emb = self.proj1(padded_x)

        pos_emb = pos_emb.permute(0, 2, 1).contiguous().view(B, num_seq * chunk_length, C)

        return pos_emb[:, :N, :]

    # def forward(self, x, H, W):
    def forward(self, x):

        pos_emb1 = self.pos_emb(x)

        # shift x
        shifted_x = torch.roll(x, shifts=self.chunk_length // 2, dims=1)
        pos_emb2 = self.pos_emb(shifted_x)
        pos_emb2 = torch.roll(pos_emb2, shifts=-self.chunk_length // 2, dims=1)

        x = x + pos_emb1 + pos_emb2

        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes, max_len):
        super(TransMIL, self).__init__()
        # net_dim = 512
        net_dim = 512
        # net_dim = 384
        # self.pos_layer = PPEG(dim=512)
        # self.pos_layer1d = PEG1D(dim=net_dim)
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        # self._fc1 = nn.Sequential(nn.Linear(1024, net_dim), nn.ReLU())
        # input_dim = 384
        input_dim = 1024
        self._fc1 = nn.Sequential(nn.Linear(input_dim, net_dim), nn.ReLU())
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, net_dim))
        self.n_classes = n_classes

        #--->translayer
        # self.pos_layer = PPEG(dim=net_dim)
        # self.pos_layer1d = PEG1D(dim=net_dim)
        # self.layer1 = TransLayer(dim=net_dim)
        # self.layer2 = TransLayer(dim=net_dim)

        # self.layer1 = TransLayerSelfAttn(dim=net_dim)
        # self.layer2 = TransLayerSelfAttn(dim=net_dim)

        self.layer1 = SelfAttnMEMLayer(dim=net_dim, use_pos=False)
        self.layer2 = SelfAttnMEMLayer(dim=net_dim, use_pos=True)

        # self.layer1 = Transformer(
        #     dim=net_dim,
        #     depth=1,
        #     heads=8,
        #     dim_head=net_dim // 8,
        #     # mlp_dim=net_dim // 8 * 4,
        #     mlp_dim=net_dim * 2,
        #     dropout=0.1
        # )

        # print(sum([p.numel() for p in self.layer1.parameters()]))
        # import sys; sys.exit()
        # self.layer2 = Transformer(
        #     dim=net_dim,
        #     depth=1,
        #     heads= 8,
        #     dim_head= net_dim // 8,
        #     mlp_dim= net_dim * 2,
        #     dropout = 0.1
        # )

        # self.layer3 = Transformer(
        #     dim= net_dim,
        #     depth=1,
        #     heads= 8,
        #     dim_head= net_dim // 8,
        #     mlp_dim= net_dim // 8 * 4,
        #     dropout = 0.1
        # )

        self.norm = nn.LayerNorm(net_dim)
        self._fc2 = nn.Linear(net_dim, self.n_classes)

        self.proj = nn.Linear(net_dim, 128)
        self.isbg = nn.Linear(net_dim, 2)

        # self.mem_length = 39
        self.mem_length = math.ceil(max_len / 1024)
        print('net_dim', net_dim, 'mem_length', self.mem_length)

        self.queues = [None for _ in range(self.n_classes)]

    # def get_queue_feats_and_labels(self):
    #     cls_feats = []
    #     cls_labels = []
    #     for cls_id, cls_feat in enumerate(self.queues):
    #         if cls_feat is None:
    #             continue

    #         cls_feats.append(cls_feat)
    #         labels = torch.zeros(cls_feat.shape[0], dtype=torch.long, device=cls_feat.device) + cls_id
    #         print('labels', labels, cls_id)
    #         cls_labels.append(labels)

    #     print('cls_feats inside method:......')
    #     print(len(cls_feats))
    #     # if len(cls_feats) > 0:
    #     #     print(cls_feats[0].shape, bool(cls_feats))
    #     if len(cls_feats) > 0:
    #         cls_feats = torch.cat(cls_feats, dim=0)
    #         cls_labels = torch.cat(cls_labels, dim=0)
    #         return cls_feats, cls_labels
    #     else:
    #         return None, None

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

    def _update_all_mems(self, mems, hids):
        res = []
        for mem, h in zip(mems, hids):
            mem = self._update_mems(mem, h.unsqueeze(dim=1))
            res.append(mem)

        return res

    def forward(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]

        # is_last = kwargs['is_last'].float() #[B, n, 1024]
        mems = kwargs.get('mems', None)

        h = self._fc1(h) #[B, n, 512]



        # ---->single layer
        #=============================================
        # if mems is not None:
        #    h = torch.cat((h, mems), dim=1)

        # #---->pad
        # H = h.shape[1]
        # _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        # add_length = _H * _W - H
        # h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        # #---->cls_token
        # B = h.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        # h = torch.cat((cls_tokens, h), dim=1)

        # # if mems is None:
        #    # mems = [None, None]

        # h = self.layer1(h) #[B, N, 512]

        # #---->PPEG
        # h = self.pos_layer(h, _H, _W) #[B, N, 512]

        # #---->PEG1D
        # # cls_tokens = h[:, 0, :].unsqueeze(dim=1)
        # # h = h[:, 1:, :]
        # # h = self.pos_layer1d(h) #[B, N, 512]
        # # h = torch.cat((cls_tokens, h), dim=1)
        # # h = self.pos_layer1d(h) #[B, N, 512]


        # h = self.layer2(h) #[B, N, 512]
        # #---->Translayer
        # mems = self._update_mems(mems, h[:, 0].unsqueeze(dim=1))



        # second layer
        # ===========================================

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        if mems is None:
           mems = [None, None]

        hids = []
        h = self.layer1(h, mem=mems[0]) #[B, N, 512]
        hids.append(h[:, 0])

        h = self.layer2(h, mems[1]) #[B, N, 512]
        hids.append(h[:, 0])

        mems = self._update_all_mems(mems, hids)


        #=============================================


        h = self.norm(h)[:,0]
        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        logits_bg = self.isbg(h)
        # if self.training:
        features = self.proj(h)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'mems': mems, 'feat': features, 'logits_bg': logits_bg}
        return results_dict



class TransMIL1(nn.Module):
    def __init__(self, n_classes, max_len):
        super(TransMIL, self).__init__()
        # net_dim = 512
        net_dim = 512
        # net_dim = 384
        # self.pos_layer = PPEG(dim=512)
        # self.pos_layer = PPEG(dim=net_dim)
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
            heads=8,
            dim_head= net_dim // 8,
            mlp_dim= net_dim * 2,
            dropout = 0.1
        )
        self.layer2 = Transformer(
            dim=net_dim,
            depth=1,
            heads= 8,
            dim_head= net_dim // 8,
            mlp_dim= net_dim * 2,
            dropout = 0.1
        )

        # self.layer3 = Transformer(
        #     dim= net_dim,
        #     depth=1,
        #     heads= 8,
        #     dim_head= net_dim // 8,
        #     mlp_dim= net_dim // 8 * 4,
        #     dropout = 0.1
        # )

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
        # self.max_len = max_len
        # self.mem_length = 39
        self.mem_length = math.ceil(max_len / 1024)
        print('net_dim', net_dim, 'mem_length', self.mem_length)

        # self.pos_emb = PositionalEmbedding(net_dim)

    def _update_all_mems(self, mems, hids):
        res = []
        for mem, h in zip(mems, hids):
            mem = self._update_mems(mem, h.unsqueeze(dim=1))
            res.append(mem)

        return res

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
        print('he.shape', h.shape)

        # is_last = kwargs['is_last'].float() #[B, n, 1024]
        mems = kwargs.get('mems', None)
        # print(h.shape, 'ccccccccccc')
        # torch.Size([2, 1024, 384])

        h = self._fc1(h) #[B, n, 512]

        if mems is not None:
            h = torch.cat((h, mems), dim=1)

        # if len(mems) == 0 :
            # mems = [None, None, None]
            # mems = [None, None]
            # mems = [
            #     torch.empty(0, dtype=h.dtype, device=h.device),
            #     torch.empty(0, dtype=h.dtype, device=h.device),
            #     torch.empty(0, dtype=h.dtype, device=h.device)
            # ]
            # mems = [None, None, None]
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        print(h.shape, cls_tokens.shape)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->PPEG
        # h = self.pos_layer(h, _H, _W) #[B, N, 512]

        # if mems[0] is None:
        #     mem_len = 0
        # else:
        #     mem_len = mems[0].shape[1]

        # klen = h.shape[1] + mem_len
        # print(h.shape)
        # pos_seq = torch.arange(klen - 1, -1, -1.0, device=h.device,
        #                            dtype=h.dtype)

        # print(pos_seq.shape, 'pos_seq')
        # pos_emb = self.pos_emb(pos_seq)
        # print(pos_emb.shape, 'pos_emb.shape', klen)
        #---->Translayer x1
        # print(h.shape, 'h.shape', mem_len)
        # h, hids = self.layer1(h, mems=mems, pos_emb=pos_emb) #[B, N, 512]
        h, hids = self.layer1(h) #[B, N, 512]

        #---->Translayer x2
        # h = self.layer2(h) #[B, N, 512]


        # h = self.layer3(h) #[B, N, 512]


        mems = self._update_mems(mems, h[:, 0].unsqueeze(dim=1))
        # mems = self._update_all_mems(mems, hids)

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
    data = torch.randn((1, 1024, 1024)).cuda()
    model = TransMIL(n_classes=2, max_len=40000).cuda()

    mems = None
    # mems = []
    print(sum([p.numel() for p in model.parameters()]))
    print(model)
    with torch.no_grad():
        for i in range(100):
            out  = model(data=data, mems=mems)
            mems = out['mems']
            print(mems.shape, 'mems')
        # for m in mems:
            # print(m.shape, 'mems')

    # print(model.eval())
    # results_dict = model(data = data)
    # print(results_dict)

    # data = torch.randn(1, 1024 + 31, 512)
    # net = PEG1D()
    # out = net(data)