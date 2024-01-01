
import sys
import os
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

import model
# from model.my_net import MyNet
from model.vit import vit_base
from model.utils import build_model

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# print(dir(dataset))
# import sys; sys.exit()
# import conf
# from pathlib import Path

# import conf
# print(conf.camlon16)





# def build_dataloader(dataset_name, img_set, dist, batch_size, num_gpus, num_workers):

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
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--ckpt_path', default='checkpoint', type=str)
    parser.add_argument('--all', action='store_true', help='if all, then return all the wsi patches in the wsi, otherwise return patches with the same label as the wsi-level label')
    parser.add_argument('--preload',  action='store_true', help='pred load training samples into RAM')
    parser.add_argument('--weights', default=None, type=str, help='pretrained weights path')

    # parser.add_argument('--model', default='mynet_A_s', type=str, help='name of the model')
    # parser.add_argument('--dataset', default='cam16')
    # parser.add_argument('--num_gpus', default=1)
    # parser.add_argument('--local_rank', default=1)
    # parser.add_argument('--epoch', type=int, default=100)
    # parser.add_argument('--ckpt_path', default='checkpoint', type=str)
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--mem_len', default=32, type=int)
    # parser.add_argument('--all', action='store_true', help='if all, then return all the wsi patches in the wsi, otherwise return patches with the same label as the wsi-level label')
    parser.add_argument('--alpha', default=-0.1, type=float)
    parser.add_argument('--debug', action='store_true', help='if print debug log')
    # parser.add_argument('--seq_len', default=256, type=int)
    parser.add_argument('--seq_len', default=512, type=int)
    parser.add_argument('--factor', default=5, type=int)


                # if not args.all:
    # parser.add_argument('--epochs', default=400, type=int)
    # parser.add_argument('--num_gpus', default=1, type=int)
    # parser.add_argument('--accum_iter', default=1, type=int,
    #                     help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    # parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
    #                     help='Name of model to train')

    # parser.add_argument('--input_size', default=224, type=int,
    #                     help='images input size')

    # parser.add_argument('--mask_ratio', default=0.75, type=float,
    #                     help='Masking ratio (percentage of removed patches).')

    # parser.add_argument('--norm_pix_loss', action='store_true',
    #                     help='Use (per-patch) normalized pixels as targets for computing loss')
    # parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    # parser.add_argument('--weight_decay', type=float, default=0.05,
    #                     help='weight decay (default: 0.05)')

    # parser.add_argument('--lr', type=float, default=None, metavar='LR',
    #                     help='learning rate (absolute lr)')
    # parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
    #                     help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0')

    # parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
    #                     help='epochs to warmup LR')

    # Dataset parameters
    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        # help='dataset path')

    # parser.add_argument('--output_dir', default='./output_dir',
                        # help='path where to save, empty for no saving')
    # parser.add_argument('--log_dir', default='./output_dir',
                        # help='path where to tensorboard log')
    # parser.add_argument('--device', default='cuda',
                        # help='device to use for training / testing')
    # parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--resume', default='',
                        # help='resume from checkpoint')

    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        # help='start epoch')
    # parser.add_argument('--num_workers', default=10, type=int)
    # parser.add_argument('--pin_mem', action='store_true',
                        # help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # parser.set_defaults(pin_mem=True)

    # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--local_rank', default=-1, type=int)
    # parser.add_argument('--dist_on_itp', action='store_true')
    # parser.add_argument('--dist_url', default='env://',
    #                     help='url used to set up distributed training')

    return parser


# def evaluate(net, ):


def main(args):
    # mics.init_process()
    # misc.init_distributed_mode(args)
    # if args.num_gpus > 1:
    # if args.num_gpus ==1:

    # print(args)
    # root_path = os.path.dirname(os.path.abspath(__file__))
    # ckpt_path = os.path.join(
    #     root_path, args.ckpt_path, TIME_NOW)
    # # log_dir = os.path.join(root_path, settings.LOG_FOLDER, args.prefix + '_' +settings.TIME_NOW)

    # if dist.get_rank() == 0:
    # if not os.path.exists(ckpt_path):
    #     print(ckpt_path)
    #     os.makedirs(ckpt_path)


    print(args)
    root_path = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(
        root_path, args.ckpt_path, TIME_NOW)
    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    # log_dir = os.path.join(root_path, settings.LOG_FOLDER, args.prefix + '_' +settings.TIME_NOW)
    log_path = os.path.join(root_path, args.log_dir, TIME_NOW)
    print(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    writer = SummaryWriter(log_dir=log_path)

    # train_dataloader = aa.utils.build_dataloader(args.dataset, 'train', dist=True, batch_size=16, num_workers=4)
    # print(dist.get_world_size())
    # train_dataloader = dataset.utils.build_dataloader(args.dataset, 'train', dist=True, batch_size=args.batch_size, num_workers=4, num_gpus=dist.get_world_size())
    train_max_len = args.seq_len * args.mem_len * args.factor
    train_dataloader = dataset.utils.build_dataloader(args.dataset, 'train', dist=dist, batch_size=args.batch_size, num_workers=4, all=args.all, drop_last=False, args=args, max_len=train_max_len)
    # val_dataloader = dataset.utils.build_dataloader(args.dataset, 'val', dist=True, batch_size=16, num_workers=4)
    # val_dataloader = dataset.utils.build_dataloader(args.dataset, 'val', dist=False, batch_size=64, num_workers=4)
    # val_dataloader = dataset.utils.build_dataloader(args.dataset, 'val', dist=dist, batch_size=128, num_workers=4, all=args.all, drop_last=False, args=args)
    val_max_len = args.seq_len * args.mem_len * 2
    val_dataloader = dataset.utils.build_dataloader(args.dataset, 'val', dist=dist, batch_size=args.batch_size, num_workers=4, all=args.all, drop_last=False, args=args, max_len=val_max_len)
    print(train_max_len, val_max_len)
    # import sys; sys.exit()
    # val_dataloader = dataset.utils.build_dataloader(args.dataset, 'val', dist=True, batch_size=16, num_workers=4, num_gpus=dist.get_world_size())
    # val_dataloader = dataset.utils.build_dataloader(args.dataset, 'val', dist=True, batch_size=16, num_workers=4, num_gpus=dist.get_world_size())
    # val_dataloader = train_dataloader
    num_classes = dataset.utils.get_num_classes(args.dataset)
    # model.vit
    # print(dir(model))
    # net = model.utils.build_model(args.model, num_classes).to(dist.get_rank())
    # net = model.utils.build_model(args.model, num_classes).cuda()
    net = model.utils.build_model(args.model, num_classes, args=args).cuda()


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

    # if args.debug:
        # from torch.profiler import profile, record_function, ProfilerActivity
        # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:


    #with torch.profiler.profile(
    #    schedule=torch.profiler.schedule(wait=1, warmup=4, active=60, repeat=1),
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test_mynet_data'),
    #    record_shapes=True,
    #    profile_memory=True,
    #    with_stack=True
    #) as prof:

    mem = None
    # mem = {
    #     'freq': None,
    #     'feat':None,
    #     'min': None
    # }
    for i in range(args.epoch):
        t1 = time.time()
        count = 0
        # net.reset()
        net.train()
        for iter_idx, data in enumerate(train_dataloader):
             # break
            #  print(count)
            #print(data)
            #img = data['img'].to(dist.get_rank())
            #label = data['label'].to(dist.get_rank())
            # print(data['img'])
            img = data['img'].cuda(non_blocking=True)
            label = data['label'].cuda(non_blocking=True)
            # print(img.shape)
            # print(img.shape)
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
            is_last = data['is_last'].cuda(non_blocking=True)

            optimizer.zero_grad()

            out, mem = net(img, mem, is_last)
            # continue
            # print(out.shape)
                        # if args.debug:
            with torch.no_grad():
               # out.soft_max()
               print(out.shape)
               pred = out.softmax(dim=1).argmax(dim=-1).detach()
               # print(pred)
               #print('pred', pred)
               #print('label', label)
               print('iter_acc', (pred == label).sum() / pred.shape[0])
               #mask = pred == label
               #l1_mask = mask & (label == 1)
               #l0_mask = mask & (label == 0)
               l1_mask = (pred == 1) & (label == 1)
               l0_mask = (pred == 0) & (label == 0)
               print('label == 1, iter_acc', l1_mask.sum() / (label == 1).sum())
               print('label == 0, iter_acc', l0_mask.sum() / (label == 0).sum())
               if mem is not None:
                   for m in mem:
                       print(m.shape)
               #print('label == 0, iter_acc', l0_mask.sum() / pred.shape[0] * 2)
            # print(out)

            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            t2 = time.time()
            print('epoch {}, iter {}, loss is {:03f}, avg time {:02f}'.format(i, iter_idx, loss, (t2 - t1) / (iter_idx + 1e-8)))
            if is_last.sum() > 0:
                print('end of training', is_last)

            # for name, param in net.named_parameters():
            #     if param.grad is None:
            #         print(name)
            # out.mean().backward()
            # print(out.shape)
            # t2 = time.time()
            # print((t2 - t1) / (data['img'].shape[0] * count))
            # print((t2 - t1) / (64 * count))

            # if iter_idx > 70:
                # break
            # prof.step()
        # break

    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            mics.visualize_lastlayer(writer, net, count)
            mics.visualize_scalar(writer, 'loss', loss, count)
            mics.visualize_scalar(writer,
                    'learning rate',
                    optimizer.param_groups[0]['lr'],
                    count)
            count += 1
            # idx_mem_iter += 1


        # import sys; sys.exit()
        mics.visualize_param_hist(
            writer,
            net,
            n_iter=i
        )

    # start eval:
        # if dist.get_rank() == 0:
        with torch.no_grad():
                # metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2).to(dist.get_rank())
                net.eval()
                metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2, average=None).cuda()
                # print(val_dataloader.shuffle)
                # import sys; sys.exit()
                count = 0
                print('evaluating.....')
                # net.reset()
                # mem = None

                # mem = {
                #     'freq': None,
                #     'feat':None,
                #     'min': None
                # }
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

                    with torch.no_grad():
                       # out.soft_max()
                       print(out.shape, iter_idx)
                       pred = out.softmax(dim=1).argmax(dim=-1).detach()
                       # print(pred)
                       #print('pred', pred)
                       #print('label', label)
                       print('iter_acc', (pred == label).sum() / pred.shape[0])
                       #mask = pred == label
                       #l1_mask = mask & (label == 1)
                       #l0_mask = mask & (label == 0)
                       l1_mask = (pred == 1) & (label == 1)
                       l0_mask = (pred == 0) & (label == 0)
                       print('label == 1, iter_acc', l1_mask.sum() / (label == 1).sum())
                       print('label == 0, iter_acc', l0_mask.sum() / (label == 0).sum())
                       print()
                    # if is_last.sum() == 0:
                    # if is_last.sum() > 0:
                    #     # print('ccccccccccccccc')
                    #     pred = out.softmax(dim=1).max(dim=1)[1]
                    #     # print(pred.device, label.device)
                    #     # print('update result')
                    #     metric.update(pred, label)

                    #     print('iter_acc', (pred == label).sum() / pred.shape[0])
                    #     mask = pred == label
                    #     l1_mask = mask & (label == 1)
                    #     l0_mask = mask & (label == 0)
                    #     print('label == 1, iter_acc', l1_mask.sum() / pred.shape[0] * 2)
                    #     print('label == 0, iter_acc', l0_mask.sum() / pred.shape[0] * 2)


                    # count += 1
                    # if count > 50:
                    #     break

        acc = metric.compute()

        # if dist.get_rank() == 0:
            # acc = 0.03
        print(f"Accuracy on all data: {acc}")

        acc_mean = acc.mean()
        # if acc > best_acc:
        if acc_mean > best_acc:
        #    # save checkpoints
            if i > 10:
                best_acc = acc_mean
                basename = '{}_{}.pt'.format(i, best_acc)
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
