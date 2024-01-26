import argparse
import csv
import json
import multiprocessing
import os
import requests
import sys
from functools import partial
from ftplib import FTP


sys.path.append(os.getcwd())

def get_files_recursive(ftp, path='', result=[]):
    try:
        # Change to the specified directory
        # print(path)
        if '.svs' not in path:
            ftp.cwd(path)
        else:
            return

        # List files in the current directory

        # Append the filenames to the result list
        # if path
        file_list = ftp.nlst()
        for filename in file_list:
            if '.svs' in filename:
                print(os.path.join(path, filename))
                result.append(os.path.join(path, filename))

        # Recursively get files in subdirectories
        for subdirectory in file_list:
            # if '.svs' not in subdirectory:
            get_files_recursive(ftp, f"{path}/{subdirectory}", result)

    except Exception as e:
        # Handle exceptions or errors if needed
        print(f"Error: {e}")


# def get_filename():
#     # with FTP('ftp://histoimage.na.icar.cnr.it/BRACS_WSI') as ftp:
#     # with FTP('histoimage.na.icar.cnr.it/BRACS_WSI') as ftp:
#     with FTP('histoimage.na.icar.cnr.it') as ftp:
#         # Login to the server
#         ftp.login('anonymous', 'your_email@example.com')
#         # ftp

#         # Change to the specified directory (default is the root directory)
#         # ftp.retrlines('LIST')
#         # path = '/BRACS'
#         ftp.cwd('BRACS_WSI/train')
#         file_list = ftp.nlst()

# # Print the list of files
#         for filename in file_list:
#             print(filename)

        # Get the list of files in the current directory
        # file_list = ftp.nlst()

def download_single_file(filename, save_dir):
    # uuid = get_uuid(filename)
    # save_path = os.path.join(save_dir, filename)
    # data_endpt = "https://api.gdc.cancer.gov/data/{}".format(uuid)
    # ftp_path = ''
    # basename = os.path.basename(filename)
    command = 'wget –no-parent -nH -c ftp://histoimage.na.icar.cnr.it{} -P {}'.format(
        filename,
        save_dir,
    )
    print(command)
    # os.system('wget -c {url}  -O {filename}'.format(url=data_endpt, filename=save_path))
    os.system(command)

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()


if '__main__' == __name__:

    # args = get_args_parser()
    # if args.dataset == 'brac':
    #     from conf.brac import settings

    # get_filename()
    # ftp =
    all_files = []
    root_directory = '/BRACS_WSI'

    with FTP('histoimage.na.icar.cnr.it') as ftp:
       # Login to the server
       ftp.login('anonymous', 'your_email@example.com')
       get_files_recursive(ftp, root_directory, all_files)

    save_dir = '/data/smb/syh/WSI_cls/bracs/img'
    # for filename in all_files:
        # print(filename)
    # filename = '/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1003691.svs'
    # download_single_file(filename=filename, save_dir=save_dir)

        # break

    downloader = partial(download_single_file, save_dir='/data/smb/syh/WSI_cls/bracs/img')
    pool = multiprocessing.Pool(processes=64)
    pool.map(downloader, all_files)
    pool.join()
