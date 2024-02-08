import argparse
import sys
import os
sys.path.append(os.getcwd())
import random

import pandas as pd
from sklearn.model_selection import KFold



def write_to_csv(dataset, fold_idx, slides, train_idx, test_idx):


    # fold_len = len(test_idx)
    val_idx = train_idx[:len(test_idx)]
    train_idx = train_idx[len(test_idx):]
    train = slides.iloc[train_idx]
    val = slides.iloc[val_idx]
    test = slides.iloc[test_idx]

    # write bool file
    csv_bool = pd.concat([
        slides['slide_id'],
        slides['slide_id'].isin(train['slide_id']),
        slides['slide_id'].isin(val['slide_id']),
        slides['slide_id'].isin(test['slide_id']),
    ], axis=1)

    csv_bool.columns = ['', 'train', 'val', 'test']

    # csv_bool.set_index('slide_id', inplace=True)
    # csv_bool.to_csv('dataset/splits/cam16/splits_{}_bool.csv'.format(0), index=False)
    # csv_bool.to_csv('dataset/splits/{}/splits_{}_bool.csv'.format(dataset, fold_idx), index=False)
    csv_bool.to_csv(
            os.path.join(
                'datasets',
                'splits',
                dataset,
                'splits_{}_bool.csv'.format(fold_idx)
            ),
        index=False)
    # print(csv_bool.index)


    # write

    train['slide_id'].reset_index(drop=True, inplace=True),
    val['slide_id'].reset_index(drop=True, inplace=True),
    test['slide_id'].reset_index(drop=True, inplace=True)
    splits =  pd.concat(
        [
            train['slide_id'],
            val['slide_id'],
            test['slide_id']
        ],
        axis=1
    )
    splits.columns = ['train', 'val', 'test']

    # splits.to_csv('dataset/splits/{}/splits_{}.csv'.format(dataset, fold_idx))
    splits.to_csv(
         os.path.join(
                'datasets',
                'splits',
                dataset,
                'splits_{}.csv'.format(fold_idx)
            ),
    )


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()

from conf.brac import settings

if __name__ == '__main__':

    args = get_args_parser()
    if args.dataset == 'brac':
        from conf.brac import settings
    elif args.dataset == 'cam16':
        from conf.camlon16 import settings
    else:
        raise ValueError('wrong dataset')

    slides = pd.read_csv(settings.file_list_csv)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(slides)):
        write_to_csv(args.dataset, fold, slides, train_idx, test_idx)