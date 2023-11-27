import torch

import glob
import os
import math
import random
# from itertools import chain
import itertools

# from . wsi import

from torch.utils.data import DataLoader
from torch.utils.data import default_collate
# import torch.distributed as dist
# from .wsi_reader import camlon16_wsis
# from conf.camlon16 import camlon16_label_fn
# from utils.utils import cycle


class WSIDataLoader:
    # DistributedSampler does not support iterabledataset
    def __init__(self,  wsis, batch_size, cls_type, num_workers=4, shuffle=True, drop_last=True, transforms=None):
        # self.data_set = data_set

        self.wsis = wsis
        self.drop_last = drop_last
        # self.num_replicas = num_gpus
        # self.dist = dist
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cls_type = cls_type
        self.shuffle = shuffle
        self.trans = transforms


        # multi gpu training
        # if dist is not None:
            # num_gpus = dist.get_world_size()


        # if self.drop_last and len(self.wsis) % self.num_replicas != 0:
        #     self.num_samples = math.ceil(
        #         (len(self.wsis) - self.num_replicas) / self.num_replicas
        # else:

        # get wsis for each gpu
        # wsis_per_gpu = self.get_subsample()

        # if dist.get_rank():
        # print(len(wsis_per_gpu), len(wsis), dist.get_rank())
        # for i in wsis_per_gpu:
            #for j in i:
                #print(j.num_patches)

        assert self.batch_size >= self.num_workers

        # split batch size for each worker
        self.bs_list = self.split_batch_size()










        # split subsamples into num_workers chunks
        # wsis_per_worker = self.split_wsis(wsis_per_gpu)
        wsis_per_worker = self.split_wsis(self.wsis)

        # for subwsi in wsis_per_worker:
        #     print(len(subwsi))
        #     print(len(subwsi), [x.num_patches for x in subwsi] )

        # for i in wsis_per_worker:
            # print([j.num_patches for j in i], dist.get_rank())

        # import sys; sys.exit()

        # each dataset only have one worker
        # wsis, batch_size, drop_last, shuffle
        # assert self.batch_size / self.num_workers
        # self.datasets = [
        #     cls_type(
        #         wsis=subset,
        #         # batch_size=len(wsis_per_worker[0]),
        #         batch_size=batch_size,
        #         drop_last=self.drop_last,
        #         # shuffle=self.shuffle,
        #         transforms=self.trans
        #     )
        #     for subset in wsis_per_worker
        # ]
        # for ss in wsis_per_worker:
            # print(len(ss))
        self.datasets = self.build_datasets(wsis_per_worker)

        # import sys; sys.exit()
        # print(self.datsaets)

        # print(datasets)


        self.dataloaders = [
            DataLoader(
                dataset,
                batch_size=None,
                num_workers=1) for dataset in self.datasets
        ]

        # rank = dist.get_ran()
        # subsample =
    # def slice_samples(self, wsis, num_chunks, rank):
    #     num_subsamples = math.ceil(len(wsis) / num_chunks)
    #     rank = self.dist.get_rank()

    #     subsample = wsis[rank * num_subsamples: (rank + 1) * num_subsamples]

    #     if len(subsample) < num_subsamples:
    #         diff = num_subsamples - len(subsample)
    #         subsample.extend(wsis[:diff])

    #     return subsample



    # def get_datasets(self):
    #     wsis_per_worker = math.ceil(len(self.sub_wsis) / self.num_workers)
    #     rank = self.dist.get_rank()
    #      = self.wsis[rank * wsis_per_worker: (rank + 1) * wsis_per_worker]

    #     # return num_samples
    def build_datasets(self, wsis):
        datasets = []
        for sub_wsis, bs in zip(wsis, self.bs_list):
            datasets.append(
                self.cls_type(
                    wsis=sub_wsis,
                    # batch_size=len(wsis_per_worker[0]),
                    batch_size=bs,
                    # drop_last=self.drop_last,
                    # shuffle=self.shuffle,
                    transforms=self.trans
            ))
            print('creating {} dataset with number of {} wsis and batch size {}'.format(
                self.cls_type, len(sub_wsis), bs
            ))

        return datasets



    def cycle(self, iterable):
        while True:
            for data in iterable:
                yield data

    def split_batch_size(self):
        base = int(self.batch_size / self.num_workers)
        bs_list = [base for _ in range(self.num_workers)]
        diff = self.batch_size - sum(bs_list)
        for idx in range(diff):
            bs_list[idx] += 1

        return bs_list


    def update_global_seq_len(self):

        # shuffle all the wsis in each dataset

        if self.shuffle:
            for dataloader in self.dataloaders:
                dataloader.dataset.shuffle()

        # get global seq across all the num_workers in current gpu
        global_seq = self.max_seq_per_gpu()
        # print(global_seq)


        # calculate the max element in seq
        outputs = []
        # for seq in zip(*[dataloader.dataset.global_seq_len for dataloader in self.dataloaders]):
        for seq in zip(*[dataloader.dataset.cal_seq_len() for dataloader in self.dataloaders]):
            # if dist.get_rank() == 0:
                # print('aaa', seq)
            outputs.append(max(seq))

        for dataset in self.datasets:
            dataset.global_seq_len = global_seq

        # print('after:', outputs)
        # for dataloader in self.dataloaders:
            # print(dataloader.dataset.global_seq_len, 'kkkkkkkkk', id(dataloader.dataset))
            # print(dataloader.dataset.global_seq_len, 'kkkkkkkkk', id(dataloader.dataset))

    # def shuffle_wsis(self):
    #     # if self.shuffle:
    #     for dataloader in self.dataloaders:
    #         dataloader.dataset.shuffle()

    #     random.shuffle(dataloaders)



    def max_seq_per_gpu(self):
        outputs = []
        # print('llllllllllllll')
        # for dataset in self.datasets:
            # print(len(dataset.wsis),  dataset.global_seq_len)
        # for seq in zip(*[dataset.global_seq_len for dataset in self.datasets]):
        for seq in zip(*[dataset.cal_seq_len() for dataset in self.datasets]):
            # if dist.get_rank() == 0:
                # print('aaa', seq)
            outputs.append(max(seq))

        return outputs


    # def cal_max_seq(self, seq_lists):



    # def update_global_seq_len(self):
    #     rank = dist.get_rank()
    #     dst_rank = 1
    #     group = dist.group.WORLD


    #     max_seq = self.max_seq_per_gpu()
    #     # print(outputs)
    #     max_seq = torch.tensor(max_seq).to(rank)
    #     # print(max_seq, rank)

    #     # how many wsis in a dataset
    #     # num_wsis = len(self.datasets[0].wsis)
    #     # num_wsis = []

    #     # gather max_seq from all gpus
    #     if rank == dst_rank:
    #         gather_list = [torch.zeros_like(max_seq) for _ in range(dist.get_world_size())]
    #     else:
    #         gather_list = None
    #     dist.gather(max_seq, gather_list=gather_list, dst=dst_rank, group=group)

    #     # calculate max_seq across all gpus
    #     if rank == dst_rank:
    #         # print(gather_list)
    #         cat_list = torch.stack(gather_list, dim=0)
    #         # print('cat_list', cat_list, cat_list.shape)
    #         max_seqs = cat_list.max(0)[0]
    #         # print('max_seq', max_seqs)

    #     # scatter max_seq accross all gpus
    #     if rank == dst_rank:
    #         scatter_list = [max_seqs for _ in range(self.num_replicas) ]
    #     else:
    #         scatter_list = None

    #     global_seq = torch.zeros_like(max_seq)
    #     dist.scatter(global_seq, scatter_list=scatter_list, src=dst_rank, group=group)



    #     #assign values to each datasets

    #     global_seq = global_seq.cpu().tolist()
    #     for dataset in self.datasets:
    #         dataset.global_seq_len = global_seq





        # global_seq = torch.zeros_like(max_seq)


        # if dist.get_rank() == dst_rank:
            # print(gather_list)




    def split_wsis(self, wsis):
        # num_samples = math.ceil(len(wsis) / self.num_workers)

        assert len(wsis) >= self.batch_size

        if self.drop_last:
            factor = math.floor(len(wsis) / self.batch_size)
            num_wsis_total = self.batch_size * factor
        else:
            factor = math.ceil(len(wsis) / self.batch_size)
            num_wsis_total = self.batch_size * factor

        # print(num_wsis_total, self.batch_size, factor)

        tmp_wsis = []
        # cycle_wsis = itertools.cycle(wsis)
        cycle_wsis = self.cycle(wsis)
        while num_wsis_total:
            tmp_wsis.append(next(cycle_wsis))
            num_wsis_total -= 1


        # print(len(tmp_wsis), self.batch_size, self.drop_last, len(wsis))
        assert len(tmp_wsis) % self.batch_size == 0

        # import sys; sys.exit()

        # print(num_samples, 'cccc', len(wsis), self.num_workers)
        # outputs = [[] for _ in range(self.num_workers)]
        outputs = []
        cum_idx = 0
        for idx in range(self.num_workers):
            # outputs.append(
                # wsis[]
            # )
            # sub_samples = wsis[idx * num_workers:(idx + 1) * self.num_workers]
            offset = self.bs_list[idx] * factor
            outputs.append(
                tmp_wsis[cum_idx:cum_idx + offset]
            )
            cum_idx += offset

        # assert cum_idx

        # for i in outputs:
            # print(len(i))

        # print(self.bs_list, 'ccc')
        return outputs
            #sub_samples = wsis[idx * num_samples:(idx + 1) * num_samples]

            ## the last chunk
            #if len(sub_samples) < num_samples:
            #    diff = num_samples - len(sub_samples)
            #    sub_samples.extend(wsis[:diff])

            #assert len(sub_samples) == num_samples

            #outputs.append(sub_samples)


        #return outputs





    # def get_subsample(self):
    #     rank = self.dist.get_rank()
    #     num_samples = math.ceil(len(self.wsis) / self.num_replicas)
    #     subsample = self.wsis[rank * num_samples: (rank + 1) * num_samples]

    #     # if dist.get_rank():
    #     #     print(subsample)

    #     # make sure each gpu acclocated the same number of wsis
    #     if len(subsample) < num_samples:
    #         # last gpu
    #         diff = num_samples - len(subsample)
    #         subsample.extend(self.wsis[:diff])

    #     return subsample


    # def collect_data(self, data_parts):
    #     pass

    def flatten(self, batch):
        outputs = []
        for each_loader in batch:
            for each_sample in each_loader:
                assert isinstance(each_sample, dict)
                outputs.append(each_sample)

        return outputs

    def __iter__(self):

        # print(self.dataloaders, self.dist.get_rank())
        self.update_global_seq_len()

        for data_parts in zip(*self.dataloaders):
            # print('data_parts', data_parts)
            # print(len(data_parts))
            # for data_part in data_parts:
                #print(data_part.keys())
                # for data in data_part:
                    # print(data)

            batch = self.flatten(data_parts)
            # print(len(batch))

            # print(len(data_parts))
            # print(len(data_parts[0]))
            # print(type(data_parts))
            # batch = list(chain.from_iterable(**list(data_parts)))
            # print(len(batch), type(batch))
            # import sys; sys.exit()
            # data_parts = default_collate(data_parts)
            batch = default_collate(batch)
            # print(data_parts.keys())
            # yield data_parts
            yield batch



        # if self.shuffle:
        #     random.shuffle(self.dataloaders)

        # for idx, wsi_path in enumerate(glob.iglob(os.path.join(wis_img_dir, '**', '*.tif'), recursive=True)):

        #       # self.wsi_filenames.append(i)
        #       # print(wsi_path, label_fn(wsi_path))
        #       # wsis.append(
        #       wsi = WSI(
        #           wsi_path,
        #           mask_path(wsi_path),
        #           patch_size=512,
        #           at_mag=5,
        #           random_rotate=True,
        #           label_fn=label_fn,
        #       )

        #       if wsi.num_patches > 0:
        #           wsis.append(wsi)

        #       else:
        #           print(wsi_path, 'is 0')
