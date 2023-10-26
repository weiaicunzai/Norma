
import os
import sys
import glob

sys.path.append(os.getcwd())
import cv2
from torchvision import transforms

# from datast.camlon16 import CAMLON16
# from dataset import CAMLON16, WSI, CAMLON16Lable
from dataset.wsi_dataset import WSIDataset
from conf.camlon16 import settings
from dataset.wsi import WSI
from dataset.dataloader import WSIDataLoader
from dataset.wsi_reader import camlon16_wsis


# mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/'
# wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/'

# mask = cv2.imread(mask_path, -1)

# # wsi = openslide.OpenSlide()

# stride = 8

# dataset = CAMLON16(wsi_path, mask_path)
# #print(dataset)
# #dataset[33]
# print(len(dataset))
# dataset[133]
def mask_path(wsi_path):
    mask_path = wsi_path.replace('training', 'training_mask')
    mask_path = mask_path.replace('.tif', '.png')
    return mask_path



def get_patch_sequqnce(dataset):

    res = []
    for wsi in dataset.wsis:
        res.append(wsi.num_patches)


    return res


def test_camlon16():

    # label_fn = CAMLON16Lable(csv_file='/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/reference.csv')

    # wsis = []
    # for idx, wsi_path in enumerate(glob.iglob(os.path.join(wis_img_dir, '**', '*.tif'), recursive=True)):

    img_size = (256, 256)

    trans = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(img_size, scale=(0.5, 2.0), ratio=(1,1)),
            # transforms.RandomCrop(img_size),
            transforms.RandomApply(
                transforms=[transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
                # p=0.3
                p=1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    count = 0
    # for filenames in settings.get_filenames('train'):
    #     count += 1

    #     # self.wsi_filenames.append(i)
    #     # print(wsi_path, label_fn(wsi_path))
    #     # wsis.append(
    #     wsi = WSI(
    #         filenames['wsi'],
    #         filenames['json'],
    #         # mask_path(wsi_path),
    #         # patch_size=512,
    #         # at_mag=5,
    #         # random_rotate=True,
    #         # label_fn=label_fn,
    #         direction=-1
    #     )

    #     if wsi.num_patches > 0:
    #         wsis.append(wsi)
    #     else:
    #         print(wsi_path, 'is 0')
    #     if count > 10:
    #         break

        # if idx == 30:
        #     break

    # import sys; sys.exit()
    # print(len(wsis))
    # for wsi in
    wsis = camlon16_wsis('train')[:40]
    dataset = WSIDataset(
        wsis=wsis,
        batch_size=4,
        transforms=trans
        )

    # seq = dataset.patch_len_seq

    # dataset.global_seq_len = seq

    # dataset.max_batch_len = 100
    import time
    start = time.time()
    count = 0
    for i in dataset:
        #  print(i)
        i[0]['img'].save('tmp1/{}.jpg'.format(count))
        count += 1
        if count > 100:
            break

    end = time.time()
    print((end - start) / count)

path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training'
# mask_dir = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/'

# wis_img_dir = ''

# test_camlon16(wis_img_dir=path)
test_camlon16()
