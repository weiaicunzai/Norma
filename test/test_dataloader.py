import os
import sys
sys.path.append(os.getcwd())


from torchvision import transforms

from dataset.wsi_reader import camlon16_wsis
from conf.camlon16 import settings
from dataset.dist_dataloader import DistWSIDataLoader
from dataset.wsi_dataset import WSIDataset
from dataset.dataloader import WSIDataLoader
# from utils.utils import init_process
# from dataset. import






def test_camlon16_num_workers():

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

    wsis = camlon16_wsis('train')

    wsis = wsis[:20]


    img_size = (256, 256)
    trans = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(img_size, scale=(0.5, 2.0), ratio=(1,1)),
            # transforms.RandomCrop(img_size),
            transforms.RandomApply(
                transforms=[transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
                p=0.3
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataloader = WSIDataLoader(
        wsis,
        batch_size=17,
        # num_gpus=3,
        cls_type=WSIDataset,
        num_workers=4,
        transforms=trans,
        drop_last=True,
    )

    # for i in
    count = 0
    import time
    start = time.time()
    for e in range(10):
        for batch in dataloader:
            # print(len(batch))
            # print(batch['img'].shape)
            # print(batch['label'].shape)
            count += 16

        print('end of each epoch')
        end = time.time()
        print((end - start) / count)


from torch.utils.data import IterableDataset, DataLoader, get_worker_info
class Test(IterableDataset):
    def __init__(self):
        self.count = 0

    def update(self):
        self.count += 1

    def __iter__(self):
        worker = get_worker_info()
        print(worker, worker.dataset.count, id(worker.dataset))
        self.count += 1
        for i in range(10):
            yield i

        print(self.count, 'in')

def test_dataset_within_dataloader():

    dataset = Test()
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1
    )

    for epoch in range(10):
        for data in dataloader:
            print(data['img'].shape)
            print(data['label'].shape)

        print(dataloader.dataset.count, 'out')
        dataloader.dataset.update()

    print(id(dataset), id(dataloader.dataset))
    print(dataset.count, dataloader.dataset.count)


    # for epoch in range(10):
    #     for i in dataset:
    #         pass


    # print(dataset.count)

test_camlon16_num_workers()
# test_dataset_within_dataloader()