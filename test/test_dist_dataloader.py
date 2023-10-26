import os
import sys
sys.path.append(os.getcwd())


from torchvision import transforms

from dataset.wsi_reader import camlon16_wsis
from conf.camlon16 import settings
from dataset.dist_dataloader import DistWSIDataLoader
from dataset.wsi_dataset import WSIDataset
from utils.utils import init_process
import torch.distributed as dist





def test():
    init_process()
    # wsis = camlon16_wsis(settings.get_filenames('train'), settings.label_fn, direction=-1)
    # print(wsis)
    #  wsis, batch_size, num_gpus, cls_type, num_workers=4, dist=None, shuffle=True
    wsis = camlon16_wsis('train')
    print(len(wsis), 'total len')
    wsis = wsis[:60]
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

    dataloader = DistWSIDataLoader(
        wsis,
        batch_size=17,
        num_gpus=3,
        cls_type=WSIDataset,
        num_workers=4,
        transforms=trans,
    )

    import time
    start = time.time()
    count = 0
    for epoch in range(10):
        for data in dataloader:
            # print(data)
            # print(data['img'].shape, data['label'].shape, data['label'], dist.get_rank())
            count += 17

            end = time.time()

            print((end - start) / count)

            pass



test()
