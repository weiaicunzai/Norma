import lmdb
import glob
import os
import json
import io
from PIL import Image
import random
import numpy as np
import cv2

from torch.utils.data import Dataset

from conf.camlon16 import settings


class CAM16(Dataset):
    def __init__(self, image_set, trans):
        super().__init__()

        if image_set == 'train':
            json_dirs = settings.train_dirs['jsons']
            lmdb_path = settings.train_dirs['lmdb'][0]
            patch_label = "/data/ssd1/by/CAMELYON16/training_json/tumor/patch_size_512_at_mag_20_patch_label"
        else:
            json_dirs = settings.test_dirs['jsons']
            lmdb_path = settings.test_dirs['lmdb'][0]
            patch_label = '/data/ssd1/by/CAMELYON16/testing/jsons/patch_size_512_at_mag_20_patch_label'


        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        # self.env = lmdb.open('/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/lmdb10000', readonly=True, lock=False)
        self.trans = trans

        self.patch_id = []

        for json_dir in json_dirs:
            for json_path in glob.iglob(os.path.join(json_dir, '**', '*.json'), recursive=True):
                # json_data = json.load(open(json_path, 'r'))
                data = json.load(open(json_path, 'r'))

                base_name = os.path.basename(json_path).replace('json', 'tif')
                for coord in data['coords'][0]:
                    # patch_id =
                    (x, y), level, (patch_size_x, patch_size_y) = coord
                    # print(x, y, level, patch_size_x, patch_size_y)
                    patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
                        basename=base_name,
                        x=x,
                        y=y,
                        level=level,
                        patch_size_x=patch_size_x,
                        patch_size_y=patch_size_y)


                    image_id = int(base_name.split('.')[0].split('_')[1])
                    # print(image_id)
                    self.patch_id.append(
                        {'name':patch_id, 'label':data['label'], 'img_id': image_id}
                    )

        # self.dummy = cv2.imread('/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/cat.jpeg')

        self.patch_labels = self.get_patch_label(patch_label)



    def get_patch_label(self, patch_label):
        res = {}
        for json_file in glob.iglob(os.path.join(patch_label, '**', '*.json'), recursive=True):

            json_data = json.load(open(json_file))
            res.update(json_data)

        return res






    def __len__(self):
        return len(self.patch_id)


    def __getitem__(self, index):
        patch_id = self.patch_id[index]

        with self.env.begin(write=False) as txn:
            img_stream = txn.get(patch_id['name'].encode())
            # mod = index % 9999
            # img_stream = txn.get('cat{:05d}.jpeg'.format(mod).encode())
            # img = Image.open(io.BytesIO(img_stream))
            # img = np.array(img)
            img = np.frombuffer(img_stream, np.uint8)
            img = cv2.imdecode(img, -1)
            # img = cv2.cvtColor(img, cv2.COLOR_)



        if self.trans:
            img = self.trans(image=img)['image']



        # if 'tumor' in patch_id:
            # patch_label = self.patch_labels[patch_id]
        # else:
            # patch_label = 0 # bg

        # print(self.patch_labels.keys())
        # print(patch_id['name'])
        patch_label = self.patch_labels.get(patch_id['name'], 0)
        # print(patch_label)
        # import sys; sys.exit()



        return {
            'img': img,
            'label': patch_label,
            # 'img_id': patch_id['img_id']
        }




class CAM16B(Dataset):
    def __init__(self, image_set, trans):
        super().__init__()

        if image_set == 'train':
            json_dirs = settings.train_dirs['jsons']
            lmdb_path = settings.train_dirs['lmdb'][0]
            patch_label = "/data/ssd1/by/CAMELYON16/training_json/tumor/patch_size_512_at_mag_20_patch_label"
        else:
            json_dirs = settings.test_dirs['jsons']
            lmdb_path = settings.test_dirs['lmdb'][0]
            patch_label = '/data/ssd1/by/CAMELYON16/testing/jsons/patch_size_512_at_mag_20_patch_label'


        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        # self.env = lmdb.open('/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/lmdb10000', readonly=True, lock=False)
        self.trans = trans
        self.image_set = image_set

        # if self.image_set == 'train':
        #     self.positive_patch_id = []
        #     self.negative_patch_id = []
        # else:
        self.patch_id = []

        for json_dir in json_dirs:
            for json_path in glob.iglob(os.path.join(json_dir, '**', '*.json'), recursive=True):
                # json_data = json.load(open(json_path, 'r'))
                data = json.load(open(json_path, 'r'))

                base_name = os.path.basename(json_path).replace('json', 'tif')
                for coord in data['coords'][0]:
                    # patch_id =
                    (x, y), level, (patch_size_x, patch_size_y) = coord
                    # print(x, y, level, patch_size_x, patch_size_y)
                    patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
                        basename=base_name,
                        x=x,
                        y=y,
                        level=level,
                        patch_size_x=patch_size_x,
                        patch_size_y=patch_size_y)


                    image_id = int(base_name.split('.')[0].split('_')[1])
                    # print(image_id)
                    # if self.image_set == 'train':
                        # if data['label']
                    self.patch_id.append(
                        {'name':patch_id, 'label':data['label'], 'img_id': image_id}
                    )

        # self.dummy = cv2.imread('/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/cat.jpeg')

        self.patch_labels = self.get_patch_label(patch_label)

        self.cancer, self.bg = self.split_label()

        self.times = 30

    def split_label(self):
        cancer = []
        bg = []
        for patch_id in self.patch_id:
            patch_label = self.patch_labels.get(patch_id['name'], 0)
            # print(type(patch_label), patch_label)
            if patch_label == 1:
                cancer.append(patch_id)
            else:
                bg.append(patch_id)

        assert len(cancer) + len(bg) == len(self.patch_id)
        return cancer, bg



    # def split_labels()

    def get_patch_label(self, patch_label):
        res = {}
        for json_file in glob.iglob(os.path.join(patch_label, '**', '*.json'), recursive=True):

            json_data = json.load(open(json_file))
            res.update(json_data)

        return res






    def __len__(self):
        if self.image_set == 'train':
            return len(self.cancer) * 2 * self.times
        else:
            return len(self.patch_id)


    def __getitem__(self, index):
        # print(self.image_set)
        if self.image_set == 'train':
            # if index < len(self.cancer) * self.times:
            if random.random() > 0.5:
                # patch_id = self.cancer[index]
                patch_id = random.choice(self.cancer)
            else:
                patch_id = random.choice(self.bg)
        else:
            patch_id = self.patch_id[index]

        # print(patch_id, 'ccccccc')
        with self.env.begin(write=False) as txn:
            img_stream = txn.get(patch_id['name'].encode())
            # mod = index % 9999
            # img_stream = txn.get('cat{:05d}.jpeg'.format(mod).encode())
            # img = Image.open(io.BytesIO(img_stream))
            # img = np.array(img)
            img = np.frombuffer(img_stream, np.uint8)
            img = cv2.imdecode(img, -1)
            # img = cv2.cvtColor(img, cv2.COLOR_)



        if self.trans:
            img = self.trans(image=img)['image']



        # if 'tumor' in patch_id:
            # patch_label = self.patch_labels[patch_id]
        # else:
            # patch_label = 0 # bg

        # print(self.patch_labels.keys())
        # print(patch_id['name'])
        patch_label = self.patch_labels.get(patch_id['name'], 0)
        # print(patch_label)
        # import sys; sys.exit()



        return {
            'img': img,
            'label': patch_label,
            # 'img_id': patch_id['img_id']
        }
