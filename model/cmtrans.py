# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
import os

import torch
import torch.nn as nn
import warnings


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        # final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat]

        return embeddings

#class MultiHeadAttentionLayer(nn.Module):
#    def __init__(self, hid_dim, n_heads, dropout, device):
#        super().__init__()
#
#        assert hid_dim % n_heads == 0
#
#        self.hid_dim = hid_dim
#        self.n_heads = n_heads
#        self.head_dim = hid_dim // n_heads
#        self.max_relative_position = 2
#
#        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
#        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)
#
#        self.fc_q = nn.Linear(hid_dim, hid_dim)
#        self.fc_k = nn.Linear(hid_dim, hid_dim)
#        self.fc_v = nn.Linear(hid_dim, hid_dim)
#
#        self.fc_o = nn.Linear(hid_dim, hid_dim)
#
#        self.dropout = nn.Dropout(dropout)
#
#        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
#
#    def forward(self, query, key, value, mask = None):
#        #query = [batch size, query len, hid dim]
#        #key = [batch size, key len, hid dim]
#        #value = [batch size, value len, hid dim]
#        batch_size = query.shape[0]
#        len_k = key.shape[1]
#        len_q = query.shape[1]
#        len_v = value.shape[1]
#
#        query = self.fc_q(query)
#        key = self.fc_k(key)
#        value = self.fc_v(value)
#
#        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))
#
#        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
#        r_k2 = self.relative_position_k(len_q, len_k)
#        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
#        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
#        attn = (attn1 + attn2) / self.scale
#
#        if mask is not None:
#            attn = attn.masked_fill(mask == 0, -1e10)
#
#        attn = self.dropout(torch.softmax(attn, dim = -1))
#
#        #attn = [batch size, n heads, query len, key len]
#        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#        weight1 = torch.matmul(attn, r_v1)
#        r_v2 = self.relative_position_v(len_q, len_v)
#        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
#        weight2 = torch.matmul(weight2, r_v2)
#        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)
#
#        x = weight1 + weight2
#
#        #x = [batch size, n heads, query len, head dim]
#
#        x = x.permute(0, 2, 1, 3).contiguous()
#
#        #x = [batch size, query len, n heads, head dim]
#
#        x = x.view(batch_size, -1, self.hid_dim)
#
#        #x = [batch size, query len, hid dim]
#
#        x = self.fc_o(x)
#
#        #x = [batch size, query len, hid dim]
#
#        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        ######################################################
        self.max_relative_position = 3

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(dim, dim, bias=qkv_bias) #
        self.fc_k = nn.Linear(dim, dim, bias=qkv_bias) #
        self.fc_v = nn.Linear(dim, dim, bias=qkv_bias) #
        ######################################################
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # print(r_k1.shape) # [B, n_heads, seq_len, dim]
        #  r_k1.permute(0, 1, 3, 2)  [B,  n_heads, dim, seq_len]
        # r_q1 :    [B, n_heads, seq_len, dim]
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) # q * v
        # print(self.n_heads, )

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        # r_q2: [len_q, batch_size * self.n_heads, head_dims]
        # r_k2: [len_q, len_k, head_dims]
        # [len_q,  num_head, head_dims] * [len_q, head_dims, len_k]
        # compute add rel pos info to each q tokens
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        # =[ len_q, num_heads, len_k]
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        # attn = self.dropout(torch.softmax(attn, dim = -1))
        attn = self.attn_drop(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.n_heads * self.head_dim)

        #x = [batch size, query len, hid dim]

        x = self.proj(x)

        #x = [batch size, query len, hid dim]
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mems=None, return_attention=False):
        if mems is not None:
            # print('mems', mems)
            # print('x', x)
            c = torch.cat([x, mems], dim=1)
        else:
            c = x

        c = self.norm1(c)
        # print(c)

        y, attn = self.attn(x, c, c)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         num_patches = (img_size // patch_size) * (img_size // patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # print(x.shape, 'before')
#         # print(self.proj(x).shape) # torch.Size([3, 384, 16, 16])
#         x = self.proj(x).flatten(2).transpose(1, 2) # torch.Size([3, 256, 384])
#         # print(x.shape)
#         return x

# class VisionTransformer(nn.Module):
class CompressiveTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, max_mem_len=32, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        # self.patch_embed = PatchEmbed(
            # img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches
        self.max_mem_len = max_mem_len

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # print(dpr)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.depth = depth
        # self.return_cls = kwargs.get('return_cls', False)

        # if self.return_cls:
            # self.fc = nn.Linear(embed_dim, 2)
        # print(kwargs, self.return_cls)
        # import sys; sys.exit()


    def attn_score(self, seq):
        """seq: [B, seq_len, dim]"""
        with torch.no_grad():
            pass


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def interpolate_pos_encoding(self, x, w, h):
    #     """interpolate positional encoding in case the input image size is different than 256x256"""
    #     npatch = x.shape[1] - 1
    #     N = self.pos_embed.shape[1] - 1
    #     if npatch == N and w == h:
    #         return self.pos_embed
    #     class_pos_embed = self.pos_embed[:, 0]
    #     patch_pos_embed = self.pos_embed[:, 1:]
    #     dim = x.shape[-1]
    #     w0 = w // self.patch_embed.patch_size
    #     h0 = h // self.patch_embed.patch_size
    #     # we add a small number to avoid floating point error in the interpolation
    #     # see discussion at https://github.com/facebookresearch/dino/issues/8
    #     w0, h0 = w0 + 0.1, h0 + 0.1
    #     # print(patch_pos_embed.shape)
    #     patch_pos_embed = nn.functional.interpolate(
    #         patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
    #         scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
    #         mode='bicubic',
    #     )
    #     # print(patch_pos_embed.shape, 'ccc')
    #     assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    #     patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    #     return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    # def prepare_tokens(self, x):
    #     B, nc, w, h = x.shape
    #     x = self.patch_embed(x)  # patch linear embedding

    #     # add the [CLS] token to the embed patch tokens
    #     # print(self.cls_token.shape)
    #     cls_tokens = self.cls_token.expand(B, -1, -1)
    #     # print(self.cls_token.shape, 'B', B)
    #     # print(cls_tokens.shape, 'B') # [3, 1, 384]
    #     # print(x.shape)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     # print(x.shape, 1111) torch.Size([3, 257, 384])
    #     # print(x[0, 0, :].mean())

    #     # add positional encoding to each token
    #     x = x + self.interpolate_pos_encoding(x, w, h)

    #     return self.pos_drop(x)

    @torch.no_grad()
    def _update_mems(self, hids, mems, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        # with torch.no_grad():
        new_mems = []
        # end_idx = mlen + max(0, qlen - 0 - self.ext_len)
        # beg_idx = max(0, end_idx - self.mem_len)
        for i in range(len(hids)):

            cat = torch.cat([mems[i], hids[i]], dim=1) # cat[1024, 512, 512]
            new_mems.append(cat[:, -mlen:, :].detach())
            # print(new_mems[-1].shape, mlen)

        return new_mems

    def init_mems(self):
        if self.max_mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.depth):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def forward(self, x, mems=None, is_last=None):
        # x = self.prepare_tokens(x)

        if mems is None:
            mems = self.init_mems()

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # print(cls_tokens.requires_grad)
        # print(self.head.weight.requires_grad)

        hids = []
        for blk, mem in zip(self.blocks, mems):
            x = blk(x, mem)
            hids.append(x[:, 0].unsqueeze(dim=1))
            # x =
        mems = self._update_mems(mems, hids, self.max_mem_len)
        x = self.norm(x)

        # print(x.shape, 'ccccccc')
        x = self.head(x)

        if is_last is not None:
            if is_last.sum() > 0:
                mems = None

        # print(x.shape)
        return x[:, 0], mems

        # return x
        # if self.return_cls:
        #     return self.fc(x[:, 0])
        # else:
        #     return x

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = CompressiveTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    # print(kwargs)
    model = CompressiveTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = CompressiveTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



# net = vit_small()
# print(net)

# img = torch.Tensor(3, 3, 256, 256)
# out = net(img)
# print(out.shape)

def get_vit256(pretrained_weights, num_classes=0, max_mem_len=32):
    r"""
    Builds ViT-256 Model.

    Args:
    - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.

    Returns:
    - model256 (torch.nn): Initialized model.
    """

    checkpoint_key = 'teacher'
    # device = torch.device("cpu")
    # model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    model256 = vit_small(patch_size=16, num_classes=num_classes, max_mem_len=max_mem_len)
    # for p in model256.parameters():
    #     p.requires_grad = False
    # model256.eval()
    # model256.cuda()

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # state_dict['blocks']
        # print(state_dict.keys())

        # for k, v in state_dict.items():
        for key in list(state_dict.keys()):
            if 'qkv' in key:
                v = state_dict[key]
                vs = torch.chunk(v, 3, dim=0)
                q = key.replace('qkv', 'fc_q')
                state_dict[q] = vs[0]

                k = key.replace('qkv', 'fc_k')
                state_dict[k] = vs[1]

                v = key.replace('qkv', 'fc_v')
                state_dict[v] = vs[2]

                del state_dict[key]



        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    return model256


# net = vit_small(num_classes=2).cuda()
#
# net = get_vit256('vit256_small_dino.pth', num_classes=2).cuda()
# net.train()

# input = torch.rand(3, 512, 384).cuda()
# mems = None
# print(net.head.weight.requires_grad)

# for i in range(10):
#     out, mems = net(input, mems)
#     # print(out.shape)