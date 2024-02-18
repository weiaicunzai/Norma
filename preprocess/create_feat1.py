import argparse
import os
import io
import csv
import struct

import torch
# from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import openslide
import time
# import multiprocessing as mp

# from multiprocessing import Process, Queue, Pool, Manager, cpu_count
# from functools import partial


import lmdb

import os
import sys
import struct
sys.path.append(os.getcwd())
import torch
from torchvision import transforms
# from PIL import Image
import  lmdb
import json
# from preprocess.utils import get_vit256
from preprocess.resnet_custom import resnet50_baseline

def eval_transforms(patch_size, pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
                     transforms.Resize((patch_size, patch_size)),
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)
    # parser.add_argument('--ckpt', required=True, default=None)
    parser.add_argument('--cont', action='store_true', help='enable')

    return parser.parse_args()

class WSI(Dataset):
    def __init__(self, path, trans):
        wsi_path, json_path = path

        self.wsi_path = wsi_path
        self.basename = os.path.basename(wsi_path)
        json_data = json.load(open(json_path))
        self.coords = json_data['coords'][0]
        self.trans = trans

        print('read {}'.format(wsi_path))
        self.wsi = openslide.OpenSlide(wsi_path)
        print('done loading {}'.format(wsi_path))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

            coord = self.coords[idx]
            (x, y), level, (patch_size_x, patch_size_y) = coord
            coord = (int(x), int(y)), int(level), (int(patch_size_x), int(patch_size_y))
            patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
                basename=self.basename,
                x=x,
                y=y,
                level=level,
                patch_size_x=patch_size_x,
                patch_size_y=patch_size_y)
            try:
                patch = self.wsi.read_region(*coord).convert('RGB')
            except Exception as e:
                print(coord, self.wsi_path)
                print(e)
                raise ValueError('some thing wrong')

            patch = self.trans(patch)
            return {'patch_id':patch_id, 'img':patch}

def get_file_path(settings):
    csv_path = settings.file_list_csv
    count = 0
    with open(csv_path, 'r') as csv_file:
        for row in csv.DictReader(csv_file):
            count += 1
            slide_id = row['slide_id']

            name = os.path.splitext(slide_id)[0]
            json_path = os.path.join(settings.json_dir, name + '.json')
            wsi_path = os.path.join(settings.wsi_dir, slide_id)

            path = wsi_path, json_path
            yield path

def create_dataloader(settings):
    datasets = []
    bs = 1024
    trans = eval_transforms(settings.patch_size, pretrained=True)
    for path in get_file_path(settings):
        datasets.append(WSI(path, trans=trans))

        # some datasets may contain over 1k or 10k wsis
        # in case we do not have enough RAM
        # we only process 32 wsis at the same time
        if len(datasets) == 32:
            yield DataLoader(ConcatDataset(datasets), num_workers=16, batch_size=bs)
            datasets = []

    # return ConcatDataset(datasets)
    if datasets:
        yield DataLoader(ConcatDataset(datasets), num_workers=16, batch_size=bs)

def write_lmdb(patch_ids, feats, settings):
    db_size = 1 << 42
    env = lmdb.open(settings.feat_dir, map_size=db_size)


    with env.begin(write=True) as txn:

        for patch_id, feat in zip(patch_ids, feats):
            # struct use 2 times less mem storage than torch.save
            # and 3 times faster to decode (struct.unpack+torch.tensor)
            # than torch.load from byte string

            feat = struct.pack('1024f', *feat)
            patch_id = patch_id.encode()
            try:
                txn.put(patch_id, feat)
            except Exception as e:
                print(e)
                raise ValueError('wrong ????')


if __name__ == '__main__':
    args = get_args_parser()
    if args.dataset == 'brac':
        from conf.brac import settings
    elif args.dataset == 'cam16':
        from conf.camlon16 import settings
    elif args.dataset == 'lung':
        from conf.lung import settings
    else:
        raise ValueError('wrong dataset')

    feat_dir = settings.feat_dir
    print(feat_dir)
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    # dataset = create_dataset(settings)
    # print(dataset)
    # bs = 1024
    # dataloader = DataLoader(dataset, num_workers=16, batch_size=bs)

    model = resnet50_baseline(pretrained=True).cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval()

    count = 0
    for dataloader in create_dataloader(settings):
        t1 = time.time()
        for idx, batch in enumerate(dataloader):

            img = batch['img'].cuda(non_blocking=True)
            with torch.no_grad():
               out = model(img)
            write_lmdb(batch['patch_id'], out.cpu().tolist(), settings)

            print(idx, 'avg batch time', (time.time() - t1) / (idx + 1), 'avg img time', (time.time() - t1) / 1024 / (idx + 1))