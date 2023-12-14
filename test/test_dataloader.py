import os
import sys
sys.path.append(os.getcwd())
from PIL import Image


from torchvision import transforms

from dataset.wsi_reader import camlon16_wsis
from conf.camlon16 import settings
from dataset.dist_dataloader import DistWSIDataLoader
from dataset.wsi_dataset import WSIDataset
from dataset.dataloader import WSIDataLoader
# from utils.utils import init_process
# from dataset. import




from torch.utils.data import Dataset

# class Test1(Dataset):
#     def __init__(self, trans):
#         super().__init__()
#         self.trans = trans

#     def __len__(self):
#         return 10000

#     def __getitem__(self, index):
#         img = Image.open('/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/cat.jpeg')
#         if self.trans:
#             img = self.trans(image=img)['image']
#         return img


# dataset = Test()
# import torch
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=17, num_workers=4)



def build_small_dataloader(img_set):
    from dataset import camlon16_wsis
    wsis = camlon16_wsis(img_set)
    # print(len(wsis))

    tmp = []
    for wsi in wsis:
        if wsi.wsi_label == 0:
            print(wsi.num_patches)
            tmp.append(wsi)

            if len(tmp) == 16:
                break

    for wsi in wsis:
        if wsi.wsi_label == 1:
            print(wsi.num_patches)
            tmp.append(wsi)

            if len(tmp) == 16 * 2:
                break


    # print(len(wsis))
    # import sys; sys.exit()

    # from dataset import WSILMDB
    wsis = tmp

    from dataset.utils import A_trans
    if img_set == 'train':
        trans = A_trans(img_set)
    else:
        trans = A_trans(img_set)

    from dataset.wsi_dataset import WSIDataset
    from dataset.dataloader import WSIDataLoader


    dataloader = WSIDataLoader(
            wsis,
            shuffle=True,
            batch_size=32,
            cls_type=WSIDataset,
            pin_memory=True,
            num_workers=4,
            transforms=trans,
            allow_repeat=False,
            drop_last=True,
        )

    # for data in dataloader:
        # print

    return dataloader




def test_camlon16_num_workers():
    from dataset.utils import build_transforms, build_dataloader

    # trans = build_transforms('train')
    #def build_dataloader(dataset_name, img_set, dist, batch_size, num_workers, num_gpus=None):

    # dataloader = build_dataloader('cam16', img_set='train', dist=False, batch_size=200, num_workers=4)
    dataloader = build_small_dataloader('train')
    print(dataloader)

    import time
    t1 = time.time()
    from viztracer import VizTracer


    # tracer = VizTracer()
    # tracer.start()
    for iter_idx, data in enumerate(dataloader):
        # print(data['is_last'])
        # print(data.keys())
        print(data['img'].shape)
        # print(data['img'].shape)
        # if iter_idx > 40:
            # break
        # print(data['label'])
        # if data['is_last'].sum() > 0:
        #     print(data)
        t2 = time.time()
        print(iter_idx, (t2 - t1) / (iter_idx + 1e-8))
        print(data['is_last'])



    # tracer.stop()
    # tracer.save('results_wsidataloader1.json')

    # count = 0
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

    # wsis = camlon16_wsis('train')

    # wsis = wsis[:20]


    #img_size = (256, 256)
    #trans = transforms.Compose([
    #        # transforms.RandomRotation(30),
    #        transforms.RandomResizedCrop(img_size, scale=(0.5, 2.0), ratio=(1,1)),
    #        # transforms.RandomApply(
    #        #     transforms=[transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
    #        #     p=0.5
    #        # ),


    #        transforms.RandomChoice
    #            (
    #                [
    #                    # nothing:
    #                    transforms.Compose([]),

    #                    # h:
    #                    transforms.RandomHorizontalFlip(p=1),

    #                    # v:
    #                    transforms.RandomVerticalFlip(p=1),

    #                    # hv:
    #                    transforms.Compose([
    #                           transforms.RandomVerticalFlip(p=1),
    #                           transforms.RandomHorizontalFlip(p=1),
    #                    ]),

    #                     #r90:
    #                    # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                    # transforms.MyRotate90(degrees=(90, 90), expand=True, p=1),
    #                    # transforms.MyRotate90(p=1),
    #                    transforms.RandomRotation(degrees=(90, 90)),

    #                    # #r90h:
    #                    transforms.Compose([
    #                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                        # transforms.MyRotate90(p=1),
    #                        transforms.RandomRotation(degrees=(90, 90)),
    #                        transforms.RandomHorizontalFlip(p=1),
    #                    ]),

    #                    # #r90v:
    #                    transforms.Compose([
    #                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                        # transforms.MyRotate90(p=1),
    #                        transforms.RandomRotation(degrees=(90, 90)),
    #                        transforms.RandomVerticalFlip(p=1),
    #                    ]),

    #                    # #r90hv:
    #                    transforms.Compose([
    #                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                        # transforms.MyRotate90(p=1),
    #                        transforms.RandomRotation(degrees=(90, 90)),
    #                        transforms.RandomHorizontalFlip(p=1),
    #                        transforms.RandomVerticalFlip(p=1),
    #                    ]),
    #                ]
    #            ),

    #        transforms.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=10),
    #        # transforms.RandomCrop(img_size),
    #        #transforms.RandomApply(
    #        #    transforms=[transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
    #        #    p=0.3
    #        #),
    #        #transforms.RandomGrayscale(p=0.1),
    #        # transforms.RandomHorizontalFlip(p=0.5),
    #        # transforms.RandomVerticalFlip(p=0.5),
    #        transforms.ToTensor(),
    #        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #])


    #import albumentations as A
    #trans = A.Compose([
    #            A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.5, 2.0), ratio=(1, 1), always_apply=True),
    #            # transforms.RandomChoice
    #            A.OneOf(
    #                [
    #                    # nothing:
    #                    A.Compose([]),

    #                    # h:
    #                    # transforms.RandomHorizontalFlip(p=1),
    #                    A.HorizontalFlip(p=1),

    #                    # v:
    #                    # transforms.RandomVerticalFlip(p=1),
    #                    A.VerticalFlip(p=1),

    #                    # hv:
    #                    # transforms.Compose([
    #                    A.Compose([
    #                           #transforms.RandomVerticalFlip(p=1),
    #                           #transforms.RandomHorizontalFlip(p=1),
    #                           A.VerticalFlip(p=1),
    #                           A.HorizontalFlip(p=1),
    #                    ]),

    #                     #r90:
    #                    # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                    # transforms.MyRotate90(degrees=(90, 90), expand=True, p=1),
    #                    # transforms.MyRotate90(p=1),
    #                    # transforms.RandomRotation(degrees=(90, 90)),
    #                    A.Rotate([90,90]),

    #                    # #r90h:
    #                    # transforms.Compose([
    #                    A.Compose([
    #                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                        # transforms.MyRotate90(p=1),
    #                        A.Rotate([90, 90]),
    #                        # transforms.RandomHorizontalFlip(p=1),
    #                        A.HorizontalFlip(p=1),
    #                    ]),

    #                    # #r90v:
    #                    # transforms.Compose([
    #                    A.Compose([
    #                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                        # transforms.MyRotate90(p=1),
    #                        # transforms.RandomRotation(degrees=(90, 90)),
    #                        A.Rotate([90, 90]),
    #                        # transforms.RandomVerticalFlip(p=1),
    #                        A.VerticalFlip(p=1)
    #                    ]),

    #                    # #r90hv:
    #                    # transforms.Compose([
    #                    A.Compose([
    #                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                        # transforms.MyRotate90(p=1),
    #                        # transforms.RandomRotation(degrees=(90, 90)),
    #                        A.Rotate([90, 90]),
    #                        #transforms.RandomHorizontalFlip(p=1),
    #                        #transforms.RandomVerticalFlip(p=1),
    #                        A.HorizontalFlip(p=1),
    #                        A.VerticalFlip(p=1),
    #                    ]),
    #                ]
    #            ),
    #            A.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4, hue=0.1, p=1),
    #            A.Compose([
    #                A.ToGray(p=1),
    #                A.ToRGB(p=1)
    #            ], p=0.1)
    #])

    ## img = Image.open('/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/cat.jpeg')
    ## img = trans(img)
    ## img.save('here.jpg')

    ## import sys; sys.exit()

    #dataloader = WSIDataLoader(
    #    wsis,
    #    batch_size=17,
    #    # num_gpus=3,
    #    cls_type=WSIDataset,
    #    num_workers=4,
    #    transforms=trans,
    #    drop_last=True,
    #)

    #import torch
    #bs = 32
    ## dataset = Test1(trans=trans)
    ## dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=4)

    ## for i in
    #count = 0
    #import time
    #start = time.time()
    #for e in range(10):
    #    for batch in dataloader:
    #        # print(len(batch))
    #        # print(batch['img'].shape)
    #        # print(batch['label'].shape)
    #        # print(batch.shape)
    #        count += bs
    #        end = time.time()
    #        print((end - start) / count)

    #    print('end of each epoch')
    #    end = time.time()
    #    print((end - start) / count)


# from torch.utils.data import IterableDataset, DataLoader, get_worker_info
# class Test(IterableDataset):
#     def __init__(self):
#         self.count = 0

#     def update(self):
#         self.count += 1

#     def __iter__(self):
#         worker = get_worker_info()
#         print(worker, worker.dataset.count, id(worker.dataset))
#         self.count += 1
#         for i in range(10):
#             yield i

#         print(self.count, 'in')

# def test_dataset_within_dataloader():

#     # dataset = Test()
#     dataloader = DataLoader(
#         dataset,
#         batch_size=None,
#         num_workers=1
#     )

#     import time
#     t1 = time.time()
#     for epoch in range(10):
#         for idx, data in enumerate(dataloader):
#             print(data['img'].shape)
#             print(data['label'].shape)
#             t2 = time.time() - t1
#             print(t2 / idx)

#         print(dataloader.dataset.count, 'out')
#         dataloader.dataset.update()

#     print(id(dataset), id(dataloader.dataset))
#     print(dataset.count, dataloader.dataset.count)


    # for epoch in range(10):
    #     for i in dataset:
    #         pass


    # print(dataset.count)

test_camlon16_num_workers()
# test_dataset_within_dataloader()