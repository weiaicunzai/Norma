import subprocess
import os


# result = subprocess.run(['conda', 'env', 'list'], stdout=subprocess.PIPE)
# conda_envs = result.stdout.decode()

# for line in conda_envs.split('\n'):
#     if '#' in line:
#         continue

#     if 'baiyu' in line:
#         continue

#     if 'songziyi' in line:
#         continue

#     if not line.strip():
#         continue

#     if not 'envs' in line:
#         continue

#     line = line.strip().split()[-1]
#     # print(line)


#     # print('cccc', line)
#     activate = os.path.dirname(line).replace('envs', 'bin/activate')
#     # subprocess.run([
#                     # "which", "conda", "&&",
#                     # "conda", 'activate', line, '&&',
#                     # 'which', 'python' ])
#     # os.system('which conda && conda activate {conda_env} && which python/'.format(
#         # conda_env=line,
#     # ))
#     # os.system('bash /data/hdd1/by/tmp_folder/run_conda.sh {}'.format(line))
#     # os.system("source {} && conda list | grep llvm && echo {} ".format(activate, line))
#     # print(line, activate)
#     print(activate)
    # os.system('bash run_conda.sh {}'.format(activate))


# a = list(range())

# import time
# dis_mem_counter = [[] for _ in range(128)]

# import numpy as np

# a = np.array((128, 64))
# print(a.shape)

# t1 = time.time()
# b = np.array((1, 64))
# c = np.concatenate([a,b], axis=0)
# # for _ in range(64):
# #     for i in range(128):
# #                 # self.dis_mem[i].append(h[i].detach())
# #     # for j in ()
# #         dis_mem_counter[i].append(
# #                     {'total':0, 'min':0}
# #                 )

# t2 = time.time()
# print(t2 - t1)
    # a = list(range(10 ** i))
    # t1 = time.time()
    # a = iter(a)
    # t2 = time.time()
    # print(t2 - t1, i)


import torch

def print_attrib(x):
    print('is_leaf', x.is_leaf, 'grad', x.grad, 'requires_grad', x.requires_grad, 'grad_fn', x.grad_fn)

x= torch.tensor([1., 2., 3.], requires_grad=True)
clone_x = x.clone()
detach_x = x.detach()



f = torch.nn.Linear(3, 1)
y = f(x)
y.backward()


# print(x.grad)

# print(clone_x.requires_grad)
# print(clone_x.grad)

# print(detach_x.requires_grad)


# a = torch.tensor([10., 10.], requires_grad=True)
# b = torch.tensor([20., 20.], requires_grad=True)

# F = a * b
# G = 2 * F

# print('a', print_attrib(a))
# print()
# print('b', print_attrib(b))
# print()
# print('F', print_attrib(F))
# print()
# print('G', print_attrib(G))

from torch.autograd import Function
# for i in dir(Function):
#     if i.startswith('__'):
#         continue
#     aa = getattr(Function, i)
#     print(i, aa)
# for i in vars(Function):
    # if i.startswith('_'):
        # continue
    # print(i)
    # print(i.)
# class LinearFunction(Function):


class MyExp(Function):

    @staticmethod
    def forward(ctx, input):
        # print(ctx, 'ccc')
        # print(type(ctx))
        result = torch.exp(input)
        ctx.save_for_backward(result) #ctx可以利用save_for_backward来保存tensors，在backward阶段可以进行获取
        return result
    @staticmethod
    def backward(ctx, grad_output):
        # print(isinstance(Function, type(ctx)))
        # print(type(ctx), type(MyExp))
        # print('ccccc')
        result, = ctx.saved_tensors
        return grad_output * result #要说明一下因为exp函数的特殊性所以reslut的导数就是reslut本身，所以这里直接是reslut

x = torch.rand(4,3,5,5, requires_grad=True)
exp = MyExp.apply # Use it by calling the apply method:
output = exp(x)
output.mean().backward()
# print(output.shape)


class MyReLU(Function):

    @staticmethod
    def forward(ctx, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        ctx.save_for_backward(input_)         # 将输入保存起来，在backward时使用
        output = input_.clamp(min=0)        # relu就是截断负数，让所有负数等于0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output
        # 因此只需求relu的导数，在乘以grad_output
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ < 0] = 0                # 上诉计算的结果就是左式。即ReLU在反向传播中可以看做一个通道选择函数，所有未达到阈值（激活值<0）的单元的梯度都为0
        return grad_input


x = torch.rand(4,3,5,5)
myrelu = MyReLU.apply # Use it by calling the apply method:
# print(myrelu, 'cccc')
# output = myrelu(x)
# print(output.shape)


class LinearFunction(Function):
    @staticmethod
    # 第一个是ctx，第二个是input，其他是可选参数。
    # 自己定义的Function中的forward()方法，所有的Variable参数将会转成tensor！因此这里的input也是tensor．在传入forward前，autograd engine会自动将Variable unpack成Tensor。
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias) # 将Tensor转变为Variable保存到ctx中
        output = input.mm(weight.t()) #.t()是转置的意思
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output) #unsqueeze(0) 扩展出第0维(原本bias只有1维)
            # expand_as(tensor)等价于expand(tensor.size()), 将原tensor按照新的size进行扩展
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        # 分别代表输入,权值,偏置三者的梯度
        # 判断三者对应的tensor是否需要进行反向求导计算梯度
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight) # 复合函数求导，链式法则
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input) #复合函数求导，链式法则
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0) #变回标量
       #sum(0)是指在维度0上面求和;squeeze(0)是指压缩维度0
       # 梯度的顺序和 forward 形参的顺序要对应
        return grad_input, grad_weight, grad_bias


# linear = LinearFunction.apply
# input = torch.randn(20,20,requires_grad=True).double()
# weight = torch.randn(20,20,requires_grad=True).double()
# bias = torch.randn(20,requires_grad=True).double()

# # from torch.autograd import gradcheck
# # test = gradcheck(LinearFunction.apply, (input,weight,bias), eps=1e-6, atol=1e-4)
# # print(test)


# a = torch.tensor((1.0), requires_grad=True)
# b = torch.tensor((2.0), requires_grad=True)

# c = a * b
# d = c + 10


# e = c.detach()
# e = a.clone()
# e = d.clone()
# print(d.data_ptr(), e.data_ptr())
# print(d, e)
# e += 1
# print(d, e)


# # d.backward()
# e.backward()
# f = d.detach()
# # print(a.grad)
# # print(e.grad, c.grad, a.grad, c.is_leaf, c.grad_fn)
# # h = a.cuda()
# # print(h.grad_fn)
# print(d.grad, e.grad, f.grad, d.grad_fn, e.grad_fn, f.grad_fn)

# print(os.getpid())
# print(os.getppid())

import seaborn as sn
import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np



# attn_score = torch.rand(32, 512) * 100


# attn_score = pd.DataFrame(attn_score.numpy().astype(np.uint8))


# print(attn_score.max(), attn_score.min())
# fig, ax = plt.subplots(figsize=(100,20))
# # hm = sn.heatmap(attn_score, vmin=0, vmax=1, xticklabels=True, yticklabels='auto', square=True, ax=ax, cbar=False)
# hm = sn.heatmap(attn_score, vmin=0, vmax=100, xticklabels=True, yticklabels='auto', square=True, ax=ax, cbar=False, annot=True)
# fig = hm.get_figure()
# fig.savefig('test.jpg', dpi=400, bbox_inches='tight', pad_inches=0.01)

def get_attn_mask(attn_score):
    # print(attn_score.min())
    alpha = -0.2
    # alpha = -1
    attn_score = attn_score - attn_score.min()
    # print('attn_score')
    # print(attn_score)
    attn_score = torch.exp(alpha * attn_score)
    # print('probs:')
    # print(attn_score)
    rand = torch.rand(size=attn_score.shape, device=attn_score.device)
    mask = attn_score > rand
    # print('rand')
    # print(rand)

    # print(attn_score)
    # print(mask)
    # print(attn_score[mask])

    return mask


    # torch.softmax(-attn_score)


# attn_score = torch.rand(10) *  10
# torch.save(attn_score, 'attn_score.pt')
# attn_score = torch.load('attn_score.pt')
# attn_score = attn_score.long()
# mask = get_attn_mask(attn_score)
# # print(attn_score)
import sys
import os
sys.path.append(os.getcwd())

from datasets.wsi import WSIJSON

from datasets.camel_data import shuffle_wsi

json_path = '/data/smb/syh/WSI_cls/cam16/json/patch_size_256_at_mag_20/normal_115.json'

wsi = WSIJSON(json_path, 0)

after_wsi = shuffle_wsi(wsi)

# def shuffle_wsi(wsi):
#     wsi = copy.deepcopy(wsi)
#     for i in range(len(wsi.coords)):
#         random.shuffle(wsi.coords[i])

#     return wsi

after_wsi = shuffle_wsi(wsi)
# for i, j in zip(wsi, after_wsi):
    # print(i, j)
