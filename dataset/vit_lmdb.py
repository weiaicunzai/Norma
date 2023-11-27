import lmdb
import glob
import os
import json
import io
from PIL import Image
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
