
import sys
import os
from typing import Any
sys.path.append(os.getcwd())

from datetime import datetime
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)


import torch
from model import vit
import argparse
from utils import mics


# from dataset import utils
# import dataset.aa as aa
import dataset
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


import model
# from model.my_net import MyNet
from model.vit import vit_base
from model.utils import build_model

import torchmetrics
import cv2


class Hooks:
    def __init__(self) -> None:
        self.result = {}

    def __call__(self, name, input):
        # self.result.update(name, input)
        self.result[name] = input


def get_args_parser():
    # parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                         help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--model', default='mynet_A_s', type=str, help='name of the model')
    parser.add_argument('--dataset', default='cam16')
    parser.add_argument('--num_gpus', default=1)
    parser.add_argument('--local_rank', default=1)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--ckpt_path', default='checkpoint', type=str)
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--mem_len', default=512, type=int)
    parser.add_argument('--all', action='store_true', help='if all, then return all the wsi patches in the wsi, otherwise return patches with the same label as the wsi-level label')

    return parser


# def evaluate(net, ):


def build_small_dataloader(img_set, args):
    from dataset import camlon16_wsis
    wsis = camlon16_wsis(img_set)
    # print(len(wsis))

    tmp = []
    for wsi in wsis:
        if wsi.wsi_label == 0:
            print(wsi.num_patches)
            # if not args.all
            tmp.append(wsi)

            if len(tmp) == 16:
                break

    for wsi in wsis:
        if wsi.wsi_label == 1:
            print(wsi.num_patches)
            tmp.append(wsi)

            if len(tmp) == 16 * 2:
                break


    if not args.all:
        for wsi in tmp:
            wsi.patch_level()


    # print(len(wsis))
    # import sys; sys.exit()

    # from dataset import WSILMDB
    # wsis
    wsis = tmp


    from dataset.utils import A_trans
    if img_set == 'train':
        trans = A_trans(img_set)
        repeats = True
    else:
        trans = A_trans(img_set)
        repeats = False

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
            allow_repeat=repeats,
            drop_last=True,
            # allow_repeat=False,
        )

    # for data in dataloader:
        # print

    return dataloader

def main(args):
    # mics.init_process()
    # misc.init_distributed_mode(args)
    # if args.num_gpus > 1:
    # if args.num_gpus ==1:

    print(args)
    root_path = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(
        root_path, args.ckpt_path, TIME_NOW)
    # log_dir = os.path.join(root_path, settings.LOG_FOLDER, args.prefix + '_' +settings.TIME_NOW)
    log_path = os.path.join(root_path, args.log_dir, TIME_NOW)
    print(log_path)
    writer = SummaryWriter(log_dir=log_path)

    # vit_weight_path = '/data/hdd1/by/tmp_folder/checkpoint/Saturday_02_December_2023_21h_55m_48s/13680_0.9084933996200562.pt'
    # if dist.get_rank() == 0:
    if not os.path.exists(ckpt_path):
        print(ckpt_path)
        os.makedirs(ckpt_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)


    # train_dataloader = aa.utils.build_dataloader(args.dataset, 'train', dist=True, batch_size=16, num_workers=4)
    # print(dist.get_world_size())
    # train_dataloader = dataset.utils.build_dataloader(args.dataset, 'train', dist=True, batch_size=args.batch_size, num_workers=4, num_gpus=dist.get_world_size())
    # A_trans
    # train_dataloader = dataset.utils.build_dataloader(args.dataset, 'train', dist=False, batch_size=args.batch_size, num_workers=4)
    # val_dataloader = dataset.utils.build_dataloader(args.dataset, 'val', dist=True, batch_size=16, num_workers=4)
    # val_dataloader = dataset.utils.build_dataloader(args.dataset, 'val', dist=False, batch_size=16, num_workers=4)
    # val_dataloader = dataset.utils.build_dataloader(args.dataset, 'val', dist=True, batch_size=16, num_workers=4, num_gpus=dist.get_world_size())
    # val_dataloader = dataset.utils.build_dataloader(args.dataset, 'val', dist=True, batch_size=16, num_workers=4, num_gpus=dist.get_world_size())
    # val_dataloader = train_dataloader
    train_dataloader = build_small_dataloader('train', args)
    # val_dataloader = build_small_dataloader('test')
    val_dataloader = build_small_dataloader('train', args)
    num_classes = dataset.utils.get_num_classes(args.dataset)
    net = model.utils.build_model(args.model, num_classes, dis_mem_len=args.mem_len).cuda()

    # for iter_idx, data in enumerate(train_dataloader):
    #     print(iter_idx)

    print(net)
    # net = net.to(dist.get_rank())
    # net = vit_base().to(dist.get_rank())
    # dpp_net =
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dist.get_rank()])
    # net = torch.nn.parallel.DistributedDataParallel(net)
    # print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = 0

    # net =
    # net =

    import time
    # time = time
    # net = MyNet(n_classes=2, n_dim=384, interval=100, dis_mem_len=64).to(dist.get_rank())
    # net = vit_base().to(dist.get_rank())
    print(sum([p.numel() for p in net.parameters()]))

    # loss_fn = torch.nn.CrossEntropyLoss()
    # import sys; sys.exit()
    # for i in range(10000):

    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    # with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test_mynet'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:

    mem = None
    hook = Hooks()
    p_label = None
    freq_mem = None
    idx_mem = None

    count = 0
    for e_idx in range(args.epoch):
        t1 = time.time()
        # net.reset()
        idx_mem_iter = 0
        for iter_idx, data in enumerate(train_dataloader):
             # break
            #  print(count)
            #print(data)
            #img = data['img'].to(dist.get_rank())
            #label = data['label'].to(dist.get_rank())
            img = data['img'].cuda(non_blocking=True)
            label = data['label'].cuda(non_blocking=True)
            # print(img.shape)
            # img = torch.randn((126, 3, 256, 256)).to(dist.get_rank())
            # img = torch.randn((16, 3, 256, 256)).to(dist.get_rank())
            # img = torch.randn((32, 3, 256, 256)).to(dist.get_rank())
            # img = torch.randn((32, 3, 256, 256))
            # img = torch.randn()
            # print(img.shape)
            # print(dist.get_rank(), img.device, net.module.vit.norm.weight.device)

            # for n, p in net.named_parameters():
            #     print('{} {} {} {}'.format(n, p.data.device, dist.get_rank(), img.device))
            # out = net.vit(img, device=img.device)
            # for name, param in net.module.named_parameters():
            #     if param.grad is None:
            #         print(name)
            # print(type(net), net.parameters())
            # print(data['is_last'])
            # is_last = data['is_last'].to(dist.get_rank())
            # if iter_idx == 180:
            #     break
            # print('wsi_label')
            # print(label)
            # print('p_label',
            # data['p_label'])
            is_last = data['is_last'].cuda(non_blocking=True)
            # hook('p_label', data['p_label'])
            out, mem = net(img, mem, is_last, hook=hook)
            # continue
            # print(out.shape)
            with torch.no_grad():
                # out.soft_max()
                pred = out.softmax(dim=1).argmax(dim=-1)
                print('pred', pred)
                print('label', label)
                print('iter_acc', (pred == label).sum() / pred.shape[0])
                mask = pred == label
                l1_mask = mask & (label == 1)
                l0_mask = mask & (label == 0)
                print('label == 1, iter_acc', l1_mask.sum() / pred.shape[0] * 2)
                print('label == 0, iter_acc', l0_mask.sum() / pred.shape[0] * 2)


            loss = loss_fn(out, label)
            loss.backward()
            # print(loss.item())
            optimizer.step()
            optimizer.zero_grad()

            # if dist.get_rank() == 0:
            t2 = time.time()
            # count += img.shape[0] * 2
            print('epoch {}, iter {}, loss is {:03f}, avg time {:02f}'.format(e_idx, iter_idx, loss, (t2 - t1) / (iter_idx + 1e-8)))


            # p_label = data['p_label'].unsqueeze(0)
            if  p_label is not None:
                p_label = torch.cat([p_label, data['p_label'].unsqueeze(1)], dim=1)
                hook('p_label', p_label.detach())
                if p_label.shape[1] > net.dis_mem_len:
                    # print(hook.result)
                    mask = hook.result['mask'].to('cpu')
                    p_label = p_label[mask.bool()].view(p_label.shape[0], -1)
                    # hook('p_label', p_label)

            else:
                p_label = data['p_label'].unsqueeze(1)
                hook('p_label', p_label.detach())

            # add freq_mem
            if freq_mem is None:
                freq_mem = torch.tensor([0] * img.shape[0]).unsqueeze(1)
            else:
                freq_mem = torch.cat([freq_mem, torch.tensor([0] * img.shape[0]).unsqueeze(1)], dim=1)

            # add to idx_mem
            if idx_mem is not None:
                idx_mem = torch.cat([idx_mem, torch.tensor([idx_mem_iter] * img.shape[0]).unsqueeze(1)], dim=1)
            else:
                idx_mem = torch.tensor([idx_mem_iter] * img.shape[0]).unsqueeze(1)
                # print(idx_mem.shape, freq_mem.shape)
                # freq_mem.scatter_add_(dim=1, index=idx_mem, src=torch.ones(idx_mem.shape, dtype=freq_mem.dtype))
                # print(freq_mem)
                # print(hook.result['mask'])

            freq_mem.scatter_add_(dim=1, index=idx_mem, src=torch.ones(idx_mem.shape, dtype=freq_mem.dtype))
            if idx_mem.shape[1] > net.dis_mem_len:
                mask = hook.result['mask'].to('cpu')
                idx_mem  = idx_mem[mask.bool()].view(idx_mem.shape[0], -1)
                # ones =

                # mask = p_label
                # new_mem = p_label[mask.bool()].view(p_label.shape[0], -1)

                # p_label[mask] = p_label[mask].view()
                # torch.cat([p_label])

            # print(hook.result.keys())


            print('-------------------')
            torch.set_printoptions(profile="full", linewidth=10000)
            print('p_label')
            print(hook.result.get('p_label'))
            print('attn_score')
            print((hook.result.get('attn_score') * 100).long())
            print('freq_mem')
            print(freq_mem)
            m = hook.result.get('attn_idx', None)
            if m is not None:
                print('attn_idx')
                print(m)
                # print((m * 100).long())

            print('label')
            print(data['label'])
            torch.set_printoptions(profile="default")


            print(hook.result['attn_score'].shape, hook.result['p_label'].shape)
            assert hook.result['attn_score'].shape == hook.result['p_label'].shape
            img = mics.draw_attn_score(
                attn_score=hook.result['attn_score'].to('cpu'),
                p_label=hook.result['p_label'].to('cpu'),
                min_index=hook.result.get('attn_idx', None),
                cell_size=5
            )
            # writer.add_image('vis_attn', img.type(torch.uint8), count, dataformats='HWC')
            cv2.imwrite('tmp2/{}.png'.format(count), cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR))

            # cv2.imwrite('tmp1/{}.png'.format(count), img.numpy())

            hook.result['attn_idx'] = None
            if is_last.sum() > 0:
                p_label = None
                idx_mem = None
                freq_mem = None
# print(x) # prints the whole tensor
                # print(attn_score.flatten())
                # torch.set_printoptions(profile="default")
            # torch.set_printoptions(profile="full")
            # print('attn_score', (hook.result['attn_score'] * 100).long())
            # print('attn_score', (hook.result['attn_score'] * 100).long())
            # print(p_label.shape)
            # for name, param in net.named_parameters():
            #     if param.grad is None:
            #         print(name)
            # out.mean().backward()
            # print(out.shape)
            # t2 = time.time()
            # print((t2 - t1) / (data['img'].shape[0] * count))
            # print((t2 - t1) / (64 * count))

            # if iter_idx > 50:
            #     break
            # prof.step()

            # visualize
            mics.visualize_lastlayer(writer, net, count)
            mics.visualize_scalar(writer, 'loss', loss, count)
            mics.visualize_scalar(writer,
                    'learning rate',
                    optimizer.param_groups[0]['lr'],
                    count)
            count += 1
            idx_mem_iter += 1
        # break

    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    # import sys; sys.exit()


    # start eval:
        # if dist.get_rank() == 0:
        with torch.no_grad():
                # metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2).to(dist.get_rank())
                metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2, average=None).cuda()
                # print(val_dataloader.shuffle)
                # import sys; sys.exit()
                count = 0
                print('evaluating.....')
                # net.reset()
                mem = None
                for data in val_dataloader:
                    # break
                    #img = data['img'].to(dist.get_rank())
                    #label = data['label'].to(dist.get_rank())
                    img = data['img'].cuda(non_blocking=True)
                    label = data['label'].cuda(non_blocking=True)
                    # img = torch.randn((126, 3, 256, 256)).to(dist.get_rank())
                    # img = torch.randn((16, 3, 256, 256)).to(dist.get_rank())
                    # img = torch.randn((32, 3, 256, 256)).to(dist.get_rank())
                    # img = torch.randn((32, 3, 256, 256))
                    # img = torch.randn()
                    # print(img.shape)
                    # print(dist.get_rank(), img.device, net.module.vit.norm.weight.device)

                    # for n, p in net.named_parameters():
                    #     print('{} {} {} {}'.format(n, p.data.device, dist.get_rank(), img.device))
                    # out = net.vit(img, device=img.device)
                    # for name, param in net.module.named_parameters():
                    #     if param.grad is None:
                    #         print(name)
                    # print(type(net), net.parameters())
                    # print(data['is_last'])
                    # is_last = data['is_last'].to(dist.get_rank())
                    is_last = data['is_last'].cuda(non_blocking=True)


                    out, mem = net(img, mem, is_last)

                    if is_last.sum() > 0:
                        # print('ccccccccccccccc')
                        pred = out.softmax(dim=1)
                        # print(pred.device, label.device)
                        print('update result')
                        print(pred, label)
                        metric.update(pred, label)



                    # count += 1
                    # if count > 50:
                    #     break

        acc = metric.compute()

        # if dist.get_rank() == 0:
            # acc = 0.03
        print(f"Accuracy on all data: {acc}")

        acc_mean = acc.mean()
        # if acc > best_acc:
        mics.visualize_metric(writer,
                        #['testB_F1', 'testB_Dice', 'testB_Haus'], testB, iter_idx)
                        'mean_acc', acc.mean(), count)

        mics.visualize_metric(writer, 'bg', acc[0], count)
        mics.visualize_metric(writer, 'cancer', acc[0], count)

        if acc_mean > best_acc:
        #    # save checkpoints
            if e_idx > 10:
                best_acc = acc_mean
                basename = '{}_{}.pt'.format(e_idx, best_acc)
                save_path = os.path.join(ckpt_path, basename)
                torch.save(net.state_dict(), os.path.join(ckpt_path, basename))
                print('saving best checkpoint to {}'.format(save_path))








    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    # device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # cudnn.benchmark = True

    # simple augmentation
    # transform_train = transforms.Compose([
    #         transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # print(dataset_train)

    # if True:  # args.distributed:
    #     num_tasks = misc.get_world_size()
    #     global_rank = misc.get_rank()
    #     sampler_train = torch.utils.data.DistributedSampler(
    #         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )
    #     print("Sampler_train = %s" % str(sampler_train))
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # if global_rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    # else:
    #     log_writer = None

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    # define the model
    # model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    # model.to(device)

    # model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    # eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    # if args.lr is None:  # only base_lr is specified
    #     args.lr = args.blr * eff_batch_size / 256

    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    # print("actual lr: %.2e" % args.lr)

    # print("accumulate grad iterations: %d" % args.accum_iter)
    # print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    # # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    # loss_scaler = NativeScaler()

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # print(f"Start training for {args.epochs} epochs")
    # start_time = time.time()
    # for epoch in range(args.start_epoch, args.epochs):
    #     if args.distributed:
    #         data_loader_train.sampler.set_epoch(epoch)
    #     train_stats = train_one_epoch(
    #         model, data_loader_train,
    #         optimizer, device, epoch, loss_scaler,
    #         log_writer=log_writer,
    #         args=args
    #     )
    #     if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
    #         misc.save_model(
    #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
    #             loss_scaler=loss_scaler, epoch=epoch)

    #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                     'epoch': epoch,}

    #     if args.output_dir and misc.is_main_process():
    #         if log_writer is not None:
    #             log_writer.flush()
    #         with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
        # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    # dataloader = dataset.utils.build_dataloader('cam16', 'train', False, batch_size=16, num_workers=4)

# net = vit.vit_small()

# count = 0
# for data in dataloader:
# print(net)
# img = torch.Tensor(3, 3, 256, 256)
# img = torch.Tensor(3, 3, 256, 256)

# discriminative_mem = []
# candidate_mem = []

# out = net(img)
# print(out.shape)
# count += 1

# if count > 10:
    # break
