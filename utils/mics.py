
import os
import torch.distributed as dist




def cycle(iterable):
    while True:
        for data in iterable:
            yield data


def init_process():
    """ Initialize the distributed environment. """
    rank = int(os.environ['LOCAL_RANK'])
    size = int(os.environ['LOCAL_WORLD_SIZE'])
    # os.environ['MASTER_PORT'] = '29500'
    # dist.init_process_group(backend, rank=rank, world_size=size)
    # print(size, '............................', os.environ["CUDA_VISIBLE_DEVICES"])
    # dist.init_process_group('nccl', rank=rank, world_size=size)
    dist.init_process_group('gloo', rank=rank, world_size=size)
    # fn(rank, size)
