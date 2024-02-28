import argparse
import math
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface, DataInterface_wsi, WSIDataModule
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='Camelyon/TransMIL.yaml',type=str)
    parser.add_argument('--gpus', default = [2])
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--dataset', required=True, type=str)
    args = parser.parse_args()
    return args

#---->main
def main(cfg, dataset, fold):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data
    # DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
    #             'train_num_workers': cfg.Data.train_dataloader.num_workers,
    #             'test_batch_size': cfg.Data.test_dataloader.batch_size,
    #             'test_num_workers': cfg.Data.test_dataloader.num_workers,
    #             'dataset_name': cfg.Data.dataset_name,
    #             'dataset_name': 'cam16',
    #             # 'dataset_cfg': cfg.Data,
    #             }

    # dm = DataInterface(**DataInterface_dict)
    # dm.train_dataset
    # dm = DataInterface_wsi(**DataInterface_dict)

    if args.dataset == 'cam16':
        from conf.camlon16 import settings
    elif args.dataset == 'brac':
        from conf.brac import settings
    elif args.dataset == 'lung':
        from conf.lung import settings
    else:
        raise ValueError('dataset value error')

    dm = WSIDataModule(
        # train_batch_size=8,
        # test_batch_size=8,
        train_batch_size=1,
        test_batch_size=1,
        settings=settings,
        fold=fold
    )

    print('datamodule', dm)

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            'settings': settings
                            }
    print(ModelInterface_dict)
    print('......................')
    model = ModelInterface(**ModelInterface_dict)
    # model = DataInterface_wsi(**ModelInterface_dict)

    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0,
        # logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        # gpus=cfg.General.gpus,
        # amp_level=cfg.General.amp_level,
        # accumulate_grad_batches=8,
        # accumulate_grad_batches=16,
        # accumulate_grad_batches=512 * 78 * 2 / (512 * 2) ,
        # accumulate_grad_batches=512 * 78 * 2 / (512 * 2) ,
        # accumulate_grad_batches=78,
        accumulate_grad_batches=math.ceil(settings.max_len / 1024),
        # accumulate_grad_batches=78,
        # accumulate_grad_batches=1,
        devices=[int(args.gpus)],
        precision=cfg.General.precision,
        # accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    #---->train or test
    if cfg.General.server == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)

    #---->update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold

    dataset = args.dataset

    #---->main
    main(cfg, dataset, args.fold)
