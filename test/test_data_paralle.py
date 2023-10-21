

import os
import random

import torch
import torch.distributed as dist


def init_process(backend='gloo'):
    """ Initialize the distributed environment. """
    rank = int(os.environ['LOCAL_RANK'])
    size = int(os.environ['LOCAL_WORLD_SIZE'])
    # os.environ['MASTER_PORT'] = '29500'
    # dist.init_process_group(backend, rank=rank, world_size=size)
    # print(size, '............................', os.environ["CUDA_VISIBLE_DEVICES"])
    dist.init_process_group('nccl', rank=rank, world_size=size)
    # fn(rank, size)


# def gather(tensor, tensor_list=None, root=0, group=None)


def run():
    init_process()
    # print(dist.get_backend())
    print(dist.get_rank())

    a = list(range(10))
    # print(random.random())
    random.shuffle(a)
    a = torch.tensor(a).to(dist.get_rank())
    print(a, dist.get_rank())


    # print(dist.group.WORLD)

    store = [torch.zeros_like(a) for _ in range(dist.get_world_size())]
    # if dist.get_rank() == 1:

    # gather every a  to store
    if dist.get_rank() == 1:
        dist.gather(a, gather_list=store, dst=1, group=dist.group.WORLD)
    else:
        dist.gather(a, dst=1, group=dist.group.WORLD)

    if dist.get_rank() == 1:
        print(store)


    b = torch.tensor(list(range(3))) + 3
    b = b.to(dist.get_rank())
    print(b)
    scatter_list = [b + i for i in range(dist.get_world_size())]


    # scatter  "scatter_list" to b
    if dist.get_rank() == 1:
        dist.scatter(b, scatter_list=scatter_list, src=1)
    else:
        dist.scatter(b, src=1)

    # if dist.get_ran() == 1:
        # print(b)

    print()
    print(dist.get_rank(), b)



run()