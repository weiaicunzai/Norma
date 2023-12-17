

import os
import random
import sys
sys.path.append(os.getcwd())

import torch
#import torch.distributed as dist
#
#
#def init_process():
#    """ Initialize the distributed environment. """
#    rank = int(os.environ['LOCAL_RANK'])
#    size = int(os.environ['LOCAL_WORLD_SIZE'])
#    # os.environ['MASTER_PORT'] = '29500'
#    # dist.init_process_group(backend, rank=rank, world_size=size)
#    # print(size, '............................', os.environ["CUDA_VISIBLE_DEVICES"])
#    dist.init_process_group('nccl', rank=rank, world_size=size)
#    # fn(rank, size)
#
#
## def gather(tensor, tensor_list=None, root=0, group=None)
#
#
#def run():
#    init_process()
#    # print(dist.get_backend())
#    print(dist.get_rank())
#
#    a = list(range(10))
#    # print(random.random())
#    random.shuffle(a)
#    a = torch.tensor(a).to(dist.get_rank())
#    print(a, dist.get_rank())
#
#
#    # print(dist.group.WORLD)
#
#    store = [torch.zeros_like(a) for _ in range(dist.get_world_size())]
#    # if dist.get_rank() == 1:
#
#    # gather every a  to store
#    if dist.get_rank() == 1:
#        dist.gather(a, gather_list=store, dst=1, group=dist.group.WORLD)
#    else:
#        dist.gather(a, dst=1, group=dist.group.WORLD)
#
#    if dist.get_rank() == 1:
#        print(store)
#
#
#    b = torch.tensor(list(range(3))) + 3
#    b = b.to(dist.get_rank())
#    print(b)
#    scatter_list = [b + i for i in range(dist.get_world_size())]
#
#
#    # scatter  "scatter_list" to b
#    if dist.get_rank() == 1:
#        dist.scatter(b, scatter_list=scatter_list, src=1)
#    else:
#        dist.scatter(b, src=1)
#
#    # if dist.get_ran() == 1:
#        # print(b)
#
#    print()
#    print(dist.get_rank(), b)
#
#
#
#run()

def torchvision_trans():

    from torchvision import transforms
    img_size=(256, 256)
    trans = transforms.Compose([
            # transforms.RandomRotation(30),
            transforms.RandomResizedCrop(img_size, scale=(0.5, 2.0), ratio=(1,1)),
            # transforms.RandomApply(
            #     transforms=[transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
            #     p=0.5
            # ),


            transforms.RandomChoice
                (
                    [
                        # nothing:
                        transforms.Compose([]),

                        # h:
                        transforms.RandomHorizontalFlip(p=1),

                        # v:
                        transforms.RandomVerticalFlip(p=1),

                        # hv:
                        transforms.Compose([
                               transforms.RandomVerticalFlip(p=1),
                               transforms.RandomHorizontalFlip(p=1),
                        ]),

                         #r90:
                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                        # transforms.MyRotate90(degrees=(90, 90), expand=True, p=1),
                        # transforms.MyRotate90(p=1),
                        transforms.RandomRotation(degrees=(90, 90)),

                        # #r90h:
                        transforms.Compose([
                            # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            # transforms.MyRotate90(p=1),
                            transforms.RandomRotation(degrees=(90, 90)),
                            transforms.RandomHorizontalFlip(p=1),
                        ]),

                        # #r90v:
                        transforms.Compose([
                            # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            # transforms.MyRotate90(p=1),
                            transforms.RandomRotation(degrees=(90, 90)),
                            transforms.RandomVerticalFlip(p=1),
                        ]),

                        # #r90hv:
                        transforms.Compose([
                            # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            # transforms.MyRotate90(p=1),
                            transforms.RandomRotation(degrees=(90, 90)),
                            transforms.RandomHorizontalFlip(p=1),
                            transforms.RandomVerticalFlip(p=1),
                        ]),
                    ]
                ),

            transforms.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=10),
            # transforms.RandomCrop(img_size),
            #transforms.RandomApply(
            #    transforms=[transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
            #    p=0.3
            #),
            #transforms.RandomGrayscale(p=0.1),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return trans

def A_trans():

    import albumentations as A
    import albumentations.pytorch as AP
    img_size = (256, 256)
    trans = A.Compose([
                A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.5, 2.0), ratio=(1, 1), always_apply=True),
                # transforms.RandomChoice
                A.OneOf(
                    [
                        # nothing:
                        A.Compose([]),

                        # h:
                        # transforms.RandomHorizontalFlip(p=1),
                        A.HorizontalFlip(p=1),

                        # v:
                        # transforms.RandomVerticalFlip(p=1),
                        A.VerticalFlip(p=1),

                        # hv:
                        # transforms.Compose([
                        A.Compose([
                               #transforms.RandomVerticalFlip(p=1),
                               #transforms.RandomHorizontalFlip(p=1),
                               A.VerticalFlip(p=1),
                               A.HorizontalFlip(p=1),
                        ]),

                         #r90:
                        # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                        # transforms.MyRotate90(degrees=(90, 90), expand=True, p=1),
                        # transforms.MyRotate90(p=1),
                        # transforms.RandomRotation(degrees=(90, 90)),
                        A.Rotate([90,90]),

                        # #r90h:
                        # transforms.Compose([
                        A.Compose([
                            # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            # transforms.MyRotate90(p=1),
                            A.Rotate([90, 90]),
                            # transforms.RandomHorizontalFlip(p=1),
                            A.HorizontalFlip(p=1),
                        ]),

                        # #r90v:
                        # transforms.Compose([
                        A.Compose([
                            # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            # transforms.MyRotate90(p=1),
                            # transforms.RandomRotation(degrees=(90, 90)),
                            A.Rotate([90, 90]),
                            # transforms.RandomVerticalFlip(p=1),
                            A.VerticalFlip(p=1)
                        ]),

                        # #r90hv:
                        # transforms.Compose([
                        A.Compose([
                            # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
                            # transforms.MyRotate90(p=1),
                            # transforms.RandomRotation(degrees=(90, 90)),
                            A.Rotate([90, 90]),
                            #transforms.RandomHorizontalFlip(p=1),
                            #transforms.RandomVerticalFlip(p=1),
                            A.HorizontalFlip(p=1),
                            A.VerticalFlip(p=1),
                        ]),
                    ]
                ),
                A.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4, hue=0.1, p=0.5),
                A.Compose([
                    A.ToGray(p=1),
                    # A.ToRGB(p=1)
                ], p=0.1),
                A.Normalize(mean=(0.62438617, 0.45624277, 0.64247613), std=(0.25213961, 0.27547218, 0.21659795)),
                AP.transforms.ToTensorV2()
    ])

    return trans

def test_lmdb_dataset():

    # import albumentations as A
    # import albumentations.pytorch as AP
    # img_size = (256, 256)
    # trans = A.Compose([
    #             A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.5, 2.0), ratio=(1, 1), always_apply=True),
    #             # transforms.RandomChoice
    #             A.OneOf(
    #                 [
    #                     # nothing:
    #                     A.Compose([]),

    #                     # h:
    #                     # transforms.RandomHorizontalFlip(p=1),
    #                     A.HorizontalFlip(p=1),

    #                     # v:
    #                     # transforms.RandomVerticalFlip(p=1),
    #                     A.VerticalFlip(p=1),

    #                     # hv:
    #                     # transforms.Compose([
    #                     A.Compose([
    #                            #transforms.RandomVerticalFlip(p=1),
    #                            #transforms.RandomHorizontalFlip(p=1),
    #                            A.VerticalFlip(p=1),
    #                            A.HorizontalFlip(p=1),
    #                     ]),

    #                      #r90:
    #                     # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                     # transforms.MyRotate90(degrees=(90, 90), expand=True, p=1),
    #                     # transforms.MyRotate90(p=1),
    #                     # transforms.RandomRotation(degrees=(90, 90)),
    #                     A.Rotate([90,90]),

    #                     # #r90h:
    #                     # transforms.Compose([
    #                     A.Compose([
    #                         # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                         # transforms.MyRotate90(p=1),
    #                         A.Rotate([90, 90]),
    #                         # transforms.RandomHorizontalFlip(p=1),
    #                         A.HorizontalFlip(p=1),
    #                     ]),

    #                     # #r90v:
    #                     # transforms.Compose([
    #                     A.Compose([
    #                         # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                         # transforms.MyRotate90(p=1),
    #                         # transforms.RandomRotation(degrees=(90, 90)),
    #                         A.Rotate([90, 90]),
    #                         # transforms.RandomVerticalFlip(p=1),
    #                         A.VerticalFlip(p=1)
    #                     ]),

    #                     # #r90hv:
    #                     # transforms.Compose([
    #                     A.Compose([
    #                         # transforms.RandomRotation(degrees=(90, 90), expand=True, p=1),
    #                         # transforms.MyRotate90(p=1),
    #                         # transforms.RandomRotation(degrees=(90, 90)),
    #                         A.Rotate([90, 90]),
    #                         #transforms.RandomHorizontalFlip(p=1),
    #                         #transforms.RandomVerticalFlip(p=1),
    #                         A.HorizontalFlip(p=1),
    #                         A.VerticalFlip(p=1),
    #                     ]),
    #                 ]
    #             ),
    #             A.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4, hue=0.1, p=0.5),
    #             A.Compose([
    #                 A.ToGray(p=1),
    #                 # A.ToRGB(p=1)
    #             ], p=0.1),
    #             A.Normalize(mean=(0.62438617, 0.45624277, 0.64247613), std=(0.25213961, 0.27547218, 0.21659795)),
    #             AP.transforms.ToTensorV2()
    # ])


    # trans=None
    # trans = torchvision_trans()
    from dataset.vit_lmdb import CAM16, CAM16B
    trans = None
    trans = A_trans()
    # dataset = CAM16('train', trans=trans)
    # dataset = CAM16('train', trans=trans)
    dataset = CAM16B('train', trans=trans)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True, pin_memory=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True, pin_memory=False)

    count = 0
    import time

    import cProfile, pstats
    from pstats import SortKey


    import cv2

    sortby = SortKey.CUMULATIVE
    count = 0
    iters = 5000 / 16
    # with cProfile.Profile() as pr:
    # for data in dataloader:
    # for
            # print(data)
    print(len(dataset))
    t1 = time.time()
    for i in range(10):
        cancer_count = 0
        bg_count = 0
        # count = 0
        for iter_idx, data in enumerate(dataloader):
        # for data in dataset:
                # count += 1
                # print(data['img'].shape)
                # print(data['label'].shape)
                # print(data['label'])
                # count += data['img'].shape[0]
            #    count += data['img'].shape[0]
            #    print(data['img'].shape)
               # break
                # cv2.imwrite('tmp/{}.jpg'.format(count), data['img'])
                #if data['label'] == 1:
                #    bg_count += 1
                #elif data['label'] == 0:
                #    cancer_count += 1
                #else:
                #    raise ValueError('fffffff')

                # cancer_count += (data['label'] == 1).sum()
                # bg_count += (data['label'] == 0).sum()
                print((time.time() - t1) / (iter_idx + 1e-8))

            # t2 = time.time()
            # count+=1
            # print('sample_time:', (t2 - t1) / count, 'iter_time:', (t2 - t1) / (count / data['img'].shape[0]))
                # if iter_idx % 100 == 0:
                    # print(cancer_count, bg_count)

            # if count == 5000:

            # if count > iters:
            #     break

        # print(bg_count, cancer_count)
        # pr.print_stats()
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps = pstats.Stats(pr).sort_stats(sortby)
        # ps.print_stats()



    #    print((t2 - t1) / count)

    # for data in dataset:
    #     count += 1
    #     t2 = time.time()
    #     print((t2 - t1) / count)



test_lmdb_dataset()