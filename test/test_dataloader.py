import argparse
import os
import sys
sys.path.append(os.getcwd())
import time


from datasets.camel_data import WSIDataset
import torch
import pandas as pd
from datasets.wsi import WSIJSON


def get_wsi(slide, json_dir):
    json_path = os.path.join(json_dir, slide + '.json')
    print('load json {}'.format(json_path))
    return WSIJSON(json_path=json_path, direction=0)

class MyWSIDataset(WSIDataset):

    def get_wsis(self, data_set, fold):
        # wsis = self.data_set()
        # from conf.camlon16 import settings
        file_list = pd.read_csv(self.settings.file_list_csv)
        file_list['slide_id'] = file_list['slide_id'].apply(
            lambda x: os.path.splitext(x)[0]
        )
        split_file = os.path.join(self.settings.split_dir, 'splits_{}.csv'.format(fold))
        splits = pd.read_csv(split_file)
        train_split = splits['train'].dropna()
        val_split = splits['val'].dropna()
        test_split = splits['test'].dropna()

        if data_set == 'train':
            mask1 = file_list['slide_id'].isin(train_split)
            mask2 = file_list['slide_id'].isin(val_split)
            slide_ids = file_list['slide_id'][mask1 | mask2]
        else:
            mask1 = file_list['slide_id'].isin(test_split)
            slide_ids = file_list['slide_id'][mask1]


        wsis = []



        # pool = mp.Pool(processes=mp.cpu_count())
        print('start load json')
        t1 = time.time()
        # pool = mp.Pool(processes=mp.cpu_count())
        # pool = mp.Pool(processes=4)
        # fn = functools.partial(get_wsi, json_dir=self.settings.json_dir)
        # wsis = pool.map(fn, slide_ids.tolist())
        wsis = []
        slide_ids = slide_ids[-10:]
        for slide_id in slide_ids:
            wsi = get_wsi(slide_id, self.settings.json_dir)
            wsis.append(wsi)

        print('done load json {}'.format(time.time() - t1))

        return wsis


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()
    if args.dataset == 'cam16':
         from conf.camlon16 import settings

    else:
         raise ValueError('wrong value')


    dataset1 = MyWSIDataset(
        settings=settings,
        data_set='train',
        fold=0,
        batch_size=8
    )

    # dataset1.orig_wsis = dataset1.orig_wsis[:10]
    dataset2 = MyWSIDataset(
        settings=settings,
        data_set='train',
        fold=0,
        batch_size=8
    )
    # dataset2.orig_wsis = dataset2.orig_wsis[:10]


    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=None, num_workers=0)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=None, num_workers=4, persistent_workers=True)


    for i in range(5):
        count = 0
        for data1, data2 in zip(dataloader1, dataloader2):
            #yield data['feat'], data['label'], data['filename'], data['is_last']
            feat1, label1, filename1, is_last1, seed1 = data1
            feat2, label2, filename2, is_last2, seed2 = data2

            # feat1 = feat1.cuda(no_blo)

            # feat1 = data1[0]
            # feat2 = data2[0]
            # print(feat1.shape)
            # print((feat1 - feat2).mean())
            count += 1
            # print((feat1 - feat2).mean())
            # print((label1 - feat2).mean())
            print('epoch {},  iter {}'.format(i, count))
            print('feat', torch.equal(feat1, feat2))
            print('label', torch.equal(label1, label2))
            # print('epoch', i, count, torch.equal(feat1, feat2))
            print('is_last', torch.equal(is_last1, is_last2))
            print('seeds', torch.equal(seed1, seed2))
            print(seed1, seed2)

            print(len(filename1) == len(filename2))
            for f1, f2 in zip(filename1, filename2):
                # print(f1 == f2)
                print(f1, f2)
