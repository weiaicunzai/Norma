import argparse
import csv
import json
import multiprocessing
import os
import requests
import sys
from functools import partial

sys.path.append(os.getcwd())


def get_uuid(filename):

    files_endpt = 'https://api.gdc.cancer.gov/files'
    # params = {'fields':'cases.submitter_id,file_id,file_name,file_size'}
    # print(filename)
    filt = {
        "op":"=",
        "content":{
            "field":"file_name",
            "value":[
                # "TCGA-3C-AALI-01Z-00-DX2.CF4496E0-AB52-4F3E-BDF5-C34833B91B7C.svs"
                filename
            ]
        }
    }
    params = {'fields':'file_name', 'filters':json.dumps(filt), }
    response = requests.get(files_endpt, params=params)

    res = response.json()

    return res['data']['hits'][0]['id']

def get_filename(settings):
    csv_file = settings.file_list_csv
    with open(csv_file, 'r') as f:
        for row in csv.DictReader(f):
            yield row['slide_id']
    # for record in settings.file_list():
        # yield os.path.basename(record['slide_id'])

def write_single_file(filename, save_dir):
    uuid = get_uuid(filename)
    save_path = os.path.join(save_dir, filename)
    data_endpt = "https://api.gdc.cancer.gov/data/{}".format(uuid)
    os.system('wget -c {url}  -O {filename}'.format(url=data_endpt, filename=save_path))

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()


if '__main__' == __name__:

    args = get_args_parser()
    if args.dataset == 'brac':
        from conf.brac import settings

    downloader = partial(write_single_file, save_dir=settings.wsi_dir)
    pool = multiprocessing.Pool(processes=100)
    pool.map(downloader, get_filename(settings))
    pool.join()
