
import os
import torch.distributed as dist
import numpy as np




# def cycle(iterable):
#     while True:
#         for data in iterable:
#             yield data


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



def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it
    Args:
        dataset: instance of torch.nn.Dataset
    Returns:
        return: mean and std of this dataset
    """

    mean = 0
    std = 0

    count = 0
    print(len(dataset), 11)
    # for img, _ in dataset:
    for img in dataset:
        img = img['img']
        mean += np.mean(img, axis=(0, 1))
        count += 1

    mean /= len(dataset)

    diff = 0
    # for img, _ in dataset:
    for img in dataset:
        img = img['img']

        diff += np.sum(np.power(img - mean, 2), axis=(0, 1))

    # N = len(dataset) * np.prod(img.shape[:2])
    N = len(dataset) * np.prod(img.size)
    std = np.sqrt(diff / N)

    mean = mean / 255
    std = std / 255

    return mean, std