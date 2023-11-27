
# import os

# import glob
# import torch.distributed as dist
# # from conf import camlon16

# def init_process():
#     """ Initialize the distributed environment. """
#     rank = int(os.environ['LOCAL_RANK'])
#     size = int(os.environ['LOCAL_WORLD_SIZE'])
#     # os.environ['MASTER_PORT'] = '29500'
#     # dist.init_process_group(backend, rank=rank, world_size=size)
#     # print(size, '............................', os.environ["CUDA_VISIBLE_DEVICES"])
#     # dist.init_process_group('nccl', rank=rank, world_size=size)
#     dist.init_process_group('gloo', rank=rank, world_size=size)
#     # fn(rank, size)


# def cycle(iterable):
#     while True:
#         for data in iterable:
#             yield data




# def camlon16_wsi_filenames(data_set):
#     outputs = []
#     if data_set == 'train':
#         dirs = camlon16.train_dirs
#     else:
#         dirs = camlon16.test_dirs


#     for wsi_dir, json_dir in zip(dirs['wsis'], dirs['jsons']):
#         for wsi_path in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
#             basename = os.path.basename(wsi_path)
#             json_path = os.path.join(json_dir, basename.replace('.tif', '.json'))
#             outputs.append({
#                 'wsi': wsi_path,
#                 'json': json_path
#             })

#     return outputs