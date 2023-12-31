from __future__ import print_function, division
import requests
import json
# import csv

import os

# import re


import multiprocessing
import argparse


# data_endpt = "https://api.gdc.cancer.gov/data"

# ids = [
#     "b658d635-258a-4f6f-8377-767a43771fe4",
#     "3968213d-b293-4b3d-8033-5b5a0ca07b6c"
#     ]

# file_name = 'TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.svs'
# params = {"file_name": file_name}

# files_endpt = 'https://api.gdc.cancer.gov/files'
# quicksearch = 'https://api.gdc.cancer.gov/v0/all'
# # params = {'fields':'cases.submitter_id,file_id,file_name,file_size'}
# filt = {
#     "op":"=",
#     "content":{
#         "field":"file_name",
#         "value":[
#             "TCGA-3C-AALI-01Z-00-DX2.CF4496E0-AB52-4F3E-BDF5-C34833B91B7C.svs"
#         ]
#     }
# }
# params = {'fields':'file_name', 'filters':json.dumps(filt), }
# response = requests.get(files_endpt, params = params)
# print(json.dumps(type(response)))
# print(type(response.json()))
# res = json.dumps(response.json(), indent=2)

# print(res['data']['hits'][(0]['id'])
# print(type(res))

# source_csv_file = '/data/hdd1/by/HIPT/2-Weakly-Supervised-Subtyping/dataset_csv/aa/tcga_brca_subset.csv.zip'
# source_csv_file = '/data/hdd1/by/HIPT/2-Weakly-Supervised-Subtyping/splits/10foldcv_subtype/tcga_brca/splits_0_bool.csv'

# source_csv_file = 'tcga_brca_subset.csv'

# import pandas as pd

# data = pd.read_csv(source_csv_file)
# print(data)

# with open('eggs.csv', newline='') as csvfile:

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

def get_filename(dataset):
    if dataset == 'brac':
        from conf.brac import settings
        for slide_id, _, _ in settings.file_list():
            yield os.path.basename(slide_id)


# def write_file(file_id, save_path):

#     # file_id = "b658d635-258a-4f6f-8377-767a43771fe4"

#     data_endpt = "https://api.gdc.cancer.gov/data/{}".format(file_id)
#     os.system('wget -c {url}  -O {filename}'.format(url=data_endpt, filename=save_path))
#     # wget.download(data_endpt, out='tmp', )

#     # print('iiii')
#     # response = requests.get(data_endpt, headers = {"Content-Type": "application/json"})
#     # print('cccccccccccc')

#     # print(response)
#     # The file name can be found in the header within the Content-Disposition key.
#     # response_head_cd = response.headers["Content-Disposition"]

#     # file_name = re.findall("filename=(.+)", response_head_cd)[0]

#     # print('writing file {}'.format(file_name))
#     # with open(file_name, "wb") as output_file:
#     #     output_file.write(response.content)

def write_single_file(filename, save_dir='/data/smb/syh/WSI_cls/TCGA_BRCA/img'):
    uuid = get_uuid(filename)
    save_path = os.path.join(save_dir, filename)
    # write_file(uuid, save_path=save_path)
    data_endpt = "https://api.gdc.cancer.gov/data/{}".format(uuid)
    os.system('wget -c {url}  -O {filename}'.format(url=data_endpt, filename=save_path))



# def write_dataset(source_csv_file, save_dir='/data/smb/syh/WSI_cls/TCGA_BRCA/img'):
#     for filename in get_filename(source_csv_file=source_csv_file):
#         print(filename)
#         uuid = get_uuid(filename)
#         save_path = os.path.join(save_dir, filename)
#         write_file(uuid, save_path=save_path)



# file_ids = list(get_filename(source_csv_file=source_csv_file))
# print(file_ids[343])
def get_args_parser():
    # parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()


if '__main__' == __name__:

    args = get_args_parser()
    pool = multiprocessing.Pool(processes=100)
    pool.map(write_single_file, get_filename(args.dataset))





# print(len(file_ids))
# pool = multiprocessing.Pool(processes=100)
# pool.map(per_write_file, file_ids)


# get_filename(source_csv_file=source_csv_file)
# write_dataset(source_csv_file=source_csv_file, save_dir='/data/smb/syh/WSI_cls/TCGA_BRCA')


# dataset = Generic_MIL_Dataset(csv_path = './dataset_csv/tcga_lung_subset.csv.zip',
# dataset = Generic_MIL_Dataset(csv_path = source_csv_file,
#                             # data_dir= os.path.join(args.data_root_dir, study_dir),
#                             data_dir= '.',
#                             mode='>>>',
#                             shuffle = False,
#                             seed = 1024,
#                             print_info = True,
#                             label_col='oncotree_code',
#                             label_dict = {'LUAD':0, 'LUSC':1},
#                             patient_strat=False,
#                             prop='fff',
#                             ignore=[])


# print(dataset)




#response = requests.post(data_endpt,
#                        data = json.dumps(params),
#                        headers={
#                            "Content-Type": "application/json"
#                            })
#
#print(response.headers)
#response_head_cd = response.headers["Content-Disposition"]
#
#file_name = re.findall("filename=(.+)", response_head_cd)[0]
#
#with open(file_name, "wb") as output_file:
#    output_file.write(response.content)