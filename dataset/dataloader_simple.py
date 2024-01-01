import pickle
import random
import warnings
# from itertools import cycle
# from utils.mics import cycle
from torch.utils.data import IterableDataset, default_collate
import torch
import cv2
from torch.utils.data import default_collate
import io
# default_collate

from PIL import Image
import lmdb
import numpy as np

import time
import math
import struct
# import json

from .wsi_reader import CAMLON16MixIn

class WSIDatasetNaive(IterableDataset):
    def __init__(self, data_set, lmdb_path, batch_size, drop_last=False, allow_reapt=False, transforms=None, dist=None):
        """the num_worker of each CAMLON16 dataset is one, """
        assert data_set in ['train', 'val']


        self.batch_size = batch_size
        self.drop_last = drop_last
        self.allow_reapt = allow_reapt
        self.data_set = data_set
        self.trans = transforms

        self.orig_wsis = self.get_wsis(data_set=data_set)
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.seed = 52

        self.dist = dist

    def split_wsis(self, wsis):
        if not self.dist.is_initialized():
            return wsis
        else:
            """get wsis for each gpu"""
            rank = self.dist.get_rank()
            num_replicas = self.dist.get_world_size()
            num_samples = math.ceil(len(wsis) / num_replicas)
            subsample = wsis[rank * num_samples: (rank + 1) * num_samples]


            # make sure each gpu acclocated the same number of wsis
            if len(subsample) < num_samples:
                diff = num_samples - len(subsample)
                subsample.extend(wsis[:diff])

            return subsample


    def get_wsis(self, data_set):
        NotImplementedError

    def set_direction(self, wsis, direction):
        for wsi in wsis:
            wsi.direction = direction

        return wsis

    def set_random_direction(self, wsis):
        random.seed(self.seed)
        for wsi in wsis:
            direction = random.randint(0, 7)
            wsi.direction = direction
        return wsis

    def orgnize_wsis(self, orig_wsis):
        wsis = []
        # when batch size is larger than the total num of wsis
        if self.batch_size > len(orig_wsis):
            if self.allow_reapt:
                wsis = orig_wsis
                while len(wsis) != self.batch_size:
                    wsis.extend(random.sample(orig_wsis, k=1))
            else:
                raise ValueError('allow_reapt should be True when batch_size is larger than the whole wsis')

        else:
            remainder = len(orig_wsis) % self.batch_size
            # if the total number of wsis is not divisible by batch_size
            if remainder > 0:
                # if we do not drop the last, we randomly select "self.batch_size - remainder" number of
                # samples add to the orig_
                random.seed(self.seed)
                if not self.drop_last:
                    wsis = orig_wsis
                    # for wsi in random.sample(self.orig_wsis, k=self.batch_size - remainder):
                    for wsi in random.sample(orig_wsis, k=self.batch_size - remainder):
                        wsis.append(wsi)

                else:
                    # if drop last, we randomly sample "total number of self.orig_wsis - remainer" number
                    # of wsis
                    wsis = random.sample(orig_wsis, k=len(orig_wsis) - remainder)
                    assert len(wsis) == len(orig_wsis) - remainder
            else:
                wsis = orig_wsis
        return wsis


    def shuffle(self, wsis):
        """manually shuffle all the wsis, because when num_workers > 0,
        the copy of dataset wont update in the main process """
        random.seed(self.seed)
        random.shuffle(wsis)

        return wsis

    def cal_seq_len(self, wsis):
        outputs = []

        for idx in range(0, len(wsis), self.batch_size):

            batch_wsi = wsis[idx : idx + self.batch_size]
            max_len = max([wsi.num_patches for wsi in batch_wsi])

            outputs.append(max_len)

        return outputs

    def cycle(self, iterable):
        while True:
            for data in iterable:
                yield data

    def read_img(self, data):

        patch_id = data['patch_id']
        with self.env.begin(write=False) as txn:
            img_stream = txn.get(patch_id.encode())
            img = np.frombuffer(img_stream, np.uint8)
            # In the case of color images, the decoded images will have the channels stored in B G R order.
            img = cv2.imdecode(img, -1)  # most time is consum
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data['img'] = img
        return data

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        # create a new list to avoid change the self.orig_wsis
        # during each epoch
        wsis = []
        for wsi in self.orig_wsis:
            wsis.append(wsi)

        wsis = self.shuffle(wsis)
        wsis = self.split_wsis(wsis) # used for ddp training
        wsis = self.orgnize_wsis(wsis)
        global_seq_len = self.cal_seq_len(wsis)

        if self.data_set != 'train':
            wsis = self.set_direction(wsis, direction=0)
            if self.drop_last:
                raise ValueError('during inference, the drop_last should not be set to true')
        else:
            wsis = self.set_random_direction(wsis)

        count = 0
        for idx in range(0, len(wsis), self.batch_size):#0

            # add seeds here to avoid different seed value for
            # different workers if we set seed += 1024 at the
            # end of each data = next(x) loop (count might not
            # be divided by each )
            self.seed += 1024

            batch_wsi = wsis[idx : idx + self.batch_size]

            assert len(batch_wsi) == self.batch_size

            batch_wsi = [self.cycle(x) for x in batch_wsi]

            max_len_idx = idx // self.batch_size

            if not global_seq_len[max_len_idx]:
                warnings.warn('max batch len equals 0')
                continue

            max_len = global_seq_len[max_len_idx]

            # if wsi len is not divisible by num_workers,
            # the last few elements will
            # change the order of reading next round
            # set a global counter to eliminate this issue
            for patch_idx in range(max_len):#104

                    outputs = []
                    for wsi in batch_wsi:

                        data = next(wsi)

                        if worker_info is not None:
                            if count % worker_info.num_workers != worker_info.id:
                                continue

                            #data['worker_id'] = worker_info.id
                            #data['count'] = self.count
                            #data['patch_idx'] = patch_idx
                            #data['seed'] = tmp_seed
                            # data['dir'] = tmp_dircts
                        data = self.read_img(data)

                        if patch_idx < max_len - 1:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        outputs.append(data)

                        if self.trans is not None:
                            data['img'] = self.trans(image=data['img'])['image'] # A

                    count += 1

                    if outputs:
                        yield default_collate(outputs)


class CAMLON16Dataset(WSIDatasetNaive, CAMLON16MixIn):
    def __init__(self, data_set, lmdb_path, batch_size, drop_last=False, allow_reapt=False, transforms=None, dist=None, all=True):
        self.all = all
        super().__init__(data_set, lmdb_path, batch_size, drop_last, allow_reapt, transforms, dist)
        # print(self.all, 'cccccccccccccccccccccccccc')

    def get_wsis(self, data_set):
        wsis = self.camlon16_wsis(data_set)
        if not self.all:
            print('hello')
            for wsi in wsis:
                wsi.patch_level()
        else:
            print('load all patches')

        return wsis


class CAMLON16DatasetFeat1(CAMLON16Dataset):
    def __init__(self, data_set, lmdb_path, batch_size, seq_len, drop_last=False, allow_reapt=False, transforms=None, dist=None, all=True):
        self.all = all
        self.seq_len = seq_len
        super().__init__(data_set, lmdb_path, batch_size, drop_last, allow_reapt, transforms, dist)

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        # create a new list to avoid change the self.orig_wsis
        # during each epoch
        wsis = []
        for wsi in self.orig_wsis:
            wsis.append(wsi)

        wsis = self.shuffle(wsis)
        wsis = self.split_wsis(wsis) # used for ddp training
        wsis = self.orgnize_wsis(wsis)
        global_seq_len = self.cal_seq_len(wsis)

        if self.data_set != 'train':
            wsis = self.set_direction(wsis, direction=0)
            if self.drop_last:
                raise ValueError('during inference, the drop_last should not be set to true')
        else:
            wsis = self.set_random_direction(wsis)

        count = 0
        for idx in range(0, len(wsis), self.batch_size):#0

            # add seeds here to avoid different seed value for
            # different workers if we set seed += 1024 at the
            # end of each data = next(x) loop (count might not
            # be divided by each )
            self.seed += 1024

            batch_wsi = wsis[idx : idx + self.batch_size]

            assert len(batch_wsi) == self.batch_size

            batch_wsi = [self.cycle(x) for x in batch_wsi]

            max_len_idx = idx // self.batch_size

            if not global_seq_len[max_len_idx]:
                warnings.warn('max batch len equals 0')
                continue

            max_len = global_seq_len[max_len_idx]

            # if wsi len is not divisible by num_workers,
            # the last few elements will
            # change the order of reading next round
            # set a global counter to eliminate this issue
            for patch_idx in range(max_len):#104

                    outputs = []
                    for wsi in batch_wsi:

                        # for _ in
                        # t1 = time.time()
                        data_list = [next(wsi) for _ in range(self.seq_len)]
                        # print(time.time() - t1)

                        if worker_info is not None:
                            if count % worker_info.num_workers != worker_info.id:
                                continue

                            #data['worker_id'] = worker_info.id
                            #data['count'] = self.count
                            #data['patch_idx'] = patch_idx
                            #data['seed'] = tmp_seed
                            # data['dir'] = tmp_dircts
                        data = self.read_img(data_list)

                        if patch_idx < max_len - 1:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        outputs.append(data)

                        # if self.trans is not None:
                            # data['img'] = self.trans(image=data['img'])['image'] # A

                    count += 1

                    if outputs:
                        # outputs
                        # print(outputs[0]['img'].shape)
                        yield default_collate(outputs)


    def concat(self, input, tensor):
        tensor = tensor.unsqueeze(dim=1)
        if input is None:
            input = tensor
        else:
            input = torch.cat([input, tensor], dim=1)
        return input

    def read_img(self, data_list):

        data = {}
        is_last = 0
        label = None
        feats = []
        for data in data_list:

            patch_id = data['patch_id']

            # if data['is_last'] == 1:
                # is_last = 1

            if label is None:
                label = data['label']

            # print(label, data['label'])
            # print(data.keys())
            assert label == data['label']

            with self.env.begin(write=False) as txn:
               img_stream = txn.get(patch_id.encode())
               feature_vector_list = struct.unpack('384f', img_stream)
               feats.append(feature_vector_list)


        data['is_last'] = is_last
        # data['img'] = torch.tensor(feats)
        data['img'] = torch.tensor(feats)
        data['label'] = label
        return data

    def cal_seq_len(self, wsis):
        outputs = []

        for idx in range(0, len(wsis), self.batch_size):

            batch_wsi = wsis[idx : idx + self.batch_size]
            max_len = max([wsi.num_patches for wsi in batch_wsi])

            assert max_len > 0

            reminder = max_len % self.seq_len
            # if max_len % self.seq_len != 0:
            if reminder != 0:
                # reminder = max_len
                max_len += self.seq_len - reminder

            outputs.append(max_len)

        return outputs



class CAMLON16DatasetFeat(CAMLON16Dataset):
    def __init__(self, data_set, lmdb_path, batch_size, seq_len, drop_last=False, allow_reapt=False, transforms=None, dist=None, all=True, preload=True, max_len=None):
        self.all = all
        self.seq_len = seq_len
        super().__init__(data_set, lmdb_path, batch_size, drop_last, allow_reapt, transforms, dist)
        # print(self.all, 'cccccccccccccccccccccccccc')

    # def concat(self, output, data):
    # def __iter__(self):
    # def flatten(self, outputs):
    #     data = {}
    #     for sample in outputs:
    #         # print(sample)
        self.preload = preload
        self.cache = {}
        if self.preload:
            self.read_samples()

        self.max_len = max_len

    def read_samples(self):
        with self.env.begin() as txn:
            # keys = list(txn.cursor().iternext(values=False))
            for iter_idx, (k, v) in enumerate(txn.cursor().iternext()):
                feature_vector = torch.load(io.BytesIO(v))
                self.cache[k.decode()] = feature_vector
                if iter_idx % 100000 == 0:
                    print('loaded {} samples'.format(iter_idx))

                # if iter+


    # def preread_batch(self, batch_wsi, )
    def read_batch(self, batch_wsi, worker_info, count):
        outputs = []
        # print('???????')
        for wsi in batch_wsi:
            data_list = [next(wsi) for _ in range(self.seq_len)]
            # print(worker_info.id, data_list)

            if worker_info is not None:
                if count % worker_info.num_workers != worker_info.id:
                    continue

            data = self.read_img(data_list)

            if worker_info is not None:
                data['worker_id'] = worker_info.id

        # print(len(data), 'cccccccccccccc')
            # print(data)
            outputs.append(data)
        return outputs

    def shift_wsi(self, batch_wsi):
        random.seed(self.seed)
        # shift 1 or self.seq_len - 1 times
        shift = random.randint(1, self.seq_len - 1)
        # print('totoal {} number of wsis , shift {} number of tokens seq_len {} '.format(len(batch_wsi), shift, self.seq_len))
        tmp = []
        for wsi in batch_wsi:

            for i in range(shift):
                next(wsi)

            tmp.append(wsi)

        return tmp, shift


    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        # create a new list to avoid change the self.orig_wsis
        # during each epoch
        wsis = []
        for wsi in self.orig_wsis:
            wsis.append(wsi)

        wsis = self.shuffle(wsis)
        wsis = self.split_wsis(wsis) # used for ddp training
        wsis = self.orgnize_wsis(wsis)
        global_seq_len = self.cal_seq_len(wsis)

        if self.data_set != 'train':
            wsis = self.set_direction(wsis, direction=0)
            if self.drop_last:
                raise ValueError('during inference, the drop_last should not be set to true')
        else:
            wsis = self.set_random_direction(wsis)

        count = 0
        # print()
        if worker_info.id == 3:
            print('all', len(self.orig_wsis), wsis)
        for idx in range(0, len(wsis), self.batch_size):#0
            # print(len(wsis), '.......................')

            # add seeds here to avoid different seed value for
            # different workers if we set seed += 1024 at the
            # end of each data = next(x) loop (count might not
            # be divided by each )
            self.seed += 1024
            batch_wsi = wsis[idx : idx + self.batch_size]
            assert len(batch_wsi) == self.batch_size
            # print(batch_wsi)
            batch_wsi = [self.cycle(x) for x in batch_wsi]
            max_len_idx = idx // self.batch_size
            if not global_seq_len[max_len_idx]:
                warnings.warn('max batch len equals 0')
                continue
            max_len = global_seq_len[max_len_idx]

            if self.max_len is not None:
                max_len = self.max_len



            assert max_len % self.seq_len == 0
            # if wsi len is not divisible by num_workers,
            # the last few elements will
            # change the order of reading next round
            # set a global counter to eliminate this issue

            batch_wsi, shift = self.shift_wsi(batch_wsi)
            # print('worker id', worker_info.id, 'shift', shift, 'batch_wsi', batch_wsi, 'seed', self.seed, 'total', len(wsis))
            # if worker_info.id == 3:
                # print('worker id', worker_info.id, 'wsis',  wsis[idx : idx + self.batch_size], 'shift', shift, 'seed', self.seed)

            # if worker_info.id == 2:
            # print('worker id', worker_info.id, 'wsis',  wsis[idx : idx + self.batch_size], 'shift', shift, 'seed', self.seed)

            for patch_idx in range(0, max_len, self.seq_len):#104
                    # print(patch_idx)

                    outputs = self.read_batch(batch_wsi, worker_info, count)
                    #outputs = []
                    #for wsi in batch_wsi:

                    #    # for _ in
                    #    # t1 = time.time()
                    #    data_list = [next(wsi) for _ in range(self.seq_len)]
                    #    # print(time.time() - t1)

                    #    if worker_info is not None:
                    #        if count % worker_info.num_workers != worker_info.id:
                    #            continue

                    #        #data['count'] = self.count
                    #        #data['patch_idx'] = patch_idx
                    #        #data['seed'] = tmp_seed
                    #        # data['dir'] = tmp_dircts
                    #    # print('------------------------------------------------')
                    #    data = self.read_img(data_list)

                    if patch_idx + self.seq_len < max_len - 1:
                        for data in outputs:
                            data['is_last'] = 0
                    else:
                        for data in outputs:
                            data['is_last'] = 1

                    # data['worker_id'] = worker_info.id
                    #    # data['max_len'] = max_len
                    #    # data['patch_idx'] = patch_idx
                    #    # data['seq_len'] = self.seq_len
                    #    # print(data['img'].shape)

                    #    outputs.append(data)

                        # if self.trans is not None:
                            # data['img'] = self.trans(image=data['img'])['image'] # A

                    # print(len(outputs), 'ccc')

                    count += 1
                    # print(count)

                    if outputs:
                        # outputs
                        # print(outputs[0]['img'].shape)
                        yield default_collate(outputs)


    def concat(self, input, tensor):
        tensor = tensor.unsqueeze(dim=1)
        if input is None:
            input = tensor
        else:
            input = torch.cat([input, tensor], dim=1)
        return input

    def read_img(self, data_list):

        data = {}
        # is_last = 0
        label = None
        feats = []
        # buffer = io.BytesIO()
        # import time
        # t1 = time.time()
        for data in data_list:

            patch_id = data['patch_id']

            # if data['is_last'] == 1:
                # is_last = 1

            if label is None:
                label = data['label']

            # print(label, data['label'])
            # print(data.keys())
            assert label == data['label']

            # print(patch_id)
            feature_vector = self.cache.get(patch_id, None)
            if not self.preload:
                if feature_vector is None:
                    with self.env.begin(write=False) as txn:
                       img_stream = txn.get(patch_id.encode())
                       feature_vector = torch.load(io.BytesIO(img_stream))
                       self.cache[patch_id] = feature_vector

            feats.append(feature_vector)

        # with self.env.begin(write=False) as txn:
            # feats = []

        # with self.env.begin(write=False) as txn:
        #     feats = [txn.get(x['patch_id'].encode()) for x in data_list]
        # print(time.time() - t1)
        # feats = [torch.load(io.BytesIO(x)) for x in feats]

        # data['is_last'] = is_last
        # data['img'] = torch.tensor(feats)
        # data['img'] = torch.tensor(feats)
        # print()
        # print(len(self.cache.keys()))
        data['img'] = torch.stack(feats, dim=0)
        data['label'] = label
        return data

    def cal_seq_len(self, wsis):
        outputs = []

        for idx in range(0, len(wsis), self.batch_size):

            batch_wsi = wsis[idx : idx + self.batch_size]
            max_len = max([wsi.num_patches for wsi in batch_wsi])

            assert max_len > 0

            reminder = max_len % self.seq_len
            # if max_len % self.seq_len != 0:
            if reminder != 0:
                # reminder = max_len
                max_len += self.seq_len - reminder

            outputs.append(max_len)

        return outputs