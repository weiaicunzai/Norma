
import io
import lmdb
from PIL import Image


def read_lmdb():
    lmdb_path = '/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/lmdb10000'
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            #print(f"Key: {key.decode()}, Value Length: {len(value)}")
            print(key.decode())
            yield value



# for v in read_lmdb():
    # img = Image.open(io.BytesIO(v))
    #print(img.size)

from torch.utils.data import Dataset
class XX(Dataset):
    def __init__(self, trans) -> None:
        super().__init__()
        self.trans = trans

    def __getitem__(self, index):
        img = Image.open('/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/cat.jpeg')
        if self.trans:
            img = self.trans(img)
        return img


dataset = XX()
import torch
dataloader = torch.utils.data.DataLoader(dataset, batch_size=17, num_workers=4)





import openslide
lmdb_path = '/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/lmdb10000'
env = lmdb.open(lmdb_path, readonly=True)
import time
t1 = time.time()
wsi_path = '/data/ssd1/by/CAMELYON16/training/normal/normal_082.tif'
wsi = openslide.OpenSlide(wsi_path)
for i in range(10000):
    # with env.begin() as txn:
    #     data = txn.get('cat09995.jpeg'.encode())
    #     img = Image.open(io.BytesIO(data))
    #     print(type(img))

    img = Image.open('/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/cat.jpeg')
    # img = wsi.read_region((3432, 3444), 5, (512, 512))
    print(i)




t2 = time.time()

print((t2 - t1) / 10000)