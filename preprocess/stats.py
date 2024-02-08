import os
import argparse
import sys
import json
import multiprocessing as mp
import statistics
sys.path.append(os.getcwd())

def get_length(json_fp):
    json_data = json.load(open(json_fp))
    coord = json_data['coords'][0]
    return len(coord)

def get_filelist(settings):
    json_dir = settings.json_dir
    for json_file in os.listdir(json_dir):
        yield os.path.join(json_dir, json_file)

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

    pool = mp.Pool(processes=mp.cpu_count())
    res = pool.map(get_length, get_filelist(settings))

    print('max', max(res))
    print('mean', statistics.mean(res))