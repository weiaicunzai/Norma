import inspect # 查看python 类的参数和模块、函数代码
import importlib # In order to dynamically import the library
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from datasets.camel_data import  WSIDataset

class DataInterface(pl.LightningDataModule):

    def __init__(self, train_batch_size=64, train_num_workers=8, test_batch_size=1, test_num_workers=1,dataset_name=None, **kwargs):
        """[summary]

        ``camel_data.py dataset``
        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """
        super().__init__()

        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        self.load_data_module()
        # self.data



    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """
        - count number of classes

        - build vocabulary

        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            #self.train_dataset = self.instancialize(state='train')
            #self.val_dataset = self.instancialize(state='val')

            self.train_dataset = self.instancialize(state='train')
            self.val_dataset = self.instancialize(state='val')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.test_dataset = self.instancialize(state='test')


    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=True)
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=0, shuffle=False)
        return DataLoader(self.train_dataset, batch_size=1, num_workers=0, shuffle=False)
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)
        # return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=0, shuffle=False)
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        # return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False)
        return DataLoader(self.test_dataset, batch_size=1, num_workers=0, shuffle=False)

    def load_data_module(self):
        camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                f'datasets.{self.dataset_name}'), camel_name)

            print(self.data_module) # datasets.camel_data.CamelData
            # import sys; sys.exit()
        except:
            raise ValueError(
                'Invalid Dataset File Name or Invalid Class Name!')

    # def load_data_module(self):
    #     # camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
    #     # try:
    #     if self.dataset_name == 'cam16':
    #         from datasets.camel_data import CAM16
    #         self.data_module = CAM16
    #     else:
    #         raise ValueError('Invalid Dataset File Name or Invalid Class Name!')


    # def instancialize(self, **other_args):
    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        # print(self.data_module, args1)
        # import sys; sys.exit()
        #return self.data_module(**args1)
        return self.data_module(**args1)



# class DataInterface_wsi(pl.LightningDataModule):
class DataInterface_wsi(pl.LightningDataModule):
    '''WSI dataset'''

    def __init__(self, train_batch_size=64, train_num_workers=8, test_batch_size=1, test_num_workers=1,dataset_name=None, **kwargs):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """
        super().__init__()

        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        self.dataset = self.load_dataset()
        # self.load_data_module()
        # self.data

    def load_dataset(self):
        if self.dataset_name == 'cam16':
            from datasets.camel_data import CAM16
            dataset = CAM16

        else:
            raise ValueError('wrong dataset names')

        return dataset



    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """
        - count number of classes

        - build vocabulary

        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        from datasets.utils import get_orig_wsis
        if self.dataset_name == 'cam16':
            from conf.camlon16 import settings

        # read all the data
        print('read all the data...')
        data = get_orig_wsis(settings)
        print('done')



        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            #self.train_dataset = self.instancialize(state='train')
            #self.val_dataset = self.instancialize(state='val')

            # self.train_dataset = self.dataset()
            self.train_dataset = self.dataset(
                 data=data,
                 data_set='train',
                 fold=0,
                 batch_size=1,
                 drop_last=False,
                 allow_reapt=False,
                 dist=None,
                 # direction=1
             )

            # self.val_dataset = self.instancialize(state='val')

            self.val_dataset = self.dataset(
                 data=data,
                 data_set='test',
                 fold=0,
                 batch_size=1,
                 drop_last=False,
                 allow_reapt=False,
                 dist=None,
                 # direction=1
             )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            # self.test_dataset = self.instancialize(state='test')
            self.test_dataset = self.dataset(
                 data=data,
                 data_set='test',
                 fold=0,
                 batch_size=1,
                 drop_last=False,
                 allow_reapt=False,
                 dist=None,
                 # direction=1
             )


    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=True)
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=0, shuffle=False)
        return DataLoader(self.train_dataset, batch_size=None, num_workers=0, shuffle=False)
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)
        # return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=0, shuffle=False)
        return DataLoader(self.val_dataset, batch_size=None, num_workers=0, shuffle=False)

    def test_dataloader(self):
        # return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False)
        return DataLoader(self.test_dataset, batch_size=None, num_workers=0, shuffle=False)

    # def load_data_module(self):
    #     camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
    #     try:
    #         self.data_module = getattr(importlib.import_module(
    #             f'datasets.{self.dataset_name}'), camel_name)

    #         print(self.data_module) # datasets.camel_data.CamelData
    #         # import sys; sys.exit()
    #     except:
    #         raise ValueError(
    #             'Invalid Dataset File Name or Invalid Class Name!')

    # def load_data_module(self):
    #     # camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
    #     # try:
    #     if self.dataset_name == 'cam16':
    #         from datasets.camel_data import CAM16
    #         self.data_module = CAM16
    #     else:
    #         raise ValueError('Invalid Dataset File Name or Invalid Class Name!')


    # def instancialize(self, **other_args):
    # def instancialize(self, **other_args):
    #     """ Instancialize a model using the corresponding parameters
    #         from self.hparams dictionary. You can also input any args
    #         to overwrite the corresponding value in self.kwargs.
    #     """
    #     class_args = inspect.getargspec(self.data_module.__init__).args[1:]
    #     inkeys = self.kwargs.keys()
    #     args1 = {}
    #     for arg in class_args:
    #         if arg in inkeys:
    #             args1[arg] = self.kwargs[arg]
    #     args1.update(other_args)
    #     return self.data_module(**args1)

class WSIDataModule(pl.LightningDataModule):
    '''WSI dataset'''

    def __init__(self, train_batch_size, test_batch_size, settings, fold):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """
        super().__init__()

        self.train_batch_size = train_batch_size
        # self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        # self.test_num_workers = test_num_workers
        # self.dataset_name = dataset_name
        self.settings = settings
        # self.dataset = self.load_dataset()
        self.dataset = WSIDataset

        self.fold = fold

    # def load_dataset(self):
    #     if self.dataset_name == 'cam16':
    #         from datasets.camel_data import CAM16
    #         dataset = CAM16

    #     else:
    #         raise ValueError('wrong dataset names')

    #     return dataset



    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """
        - count number of classes

        - build vocabulary

        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        # from datasets.utils import get_orig_wsis
        # if self.dataset_name == 'cam16':
        #     from conf.camlon16 import settings

        # read all the data
        # print('read all the data...')
        # data = get_orig_wsis(settings)
        # print('done')



        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            #self.train_dataset = self.instancialize(state='train')
            #self.val_dataset = self.instancialize(state='val')

            # self.train_dataset = self.dataset()
            self.train_dataset = self.dataset(
                settings=self.settings,
                data_set='train',
                fold=self.fold,
                batch_size=self.train_batch_size,
                drop_last=False,
                allow_reapt=False,
                dist=None,
                 # direction=1
             )

            # self.val_dataset = self.instancialize(state='val')

            self.val_dataset = self.dataset(
                 settings=self.settings,
                 data_set='test',
                 fold=self.fold,
                 batch_size=self.train_batch_size,
                 drop_last=False,
                 allow_reapt=False,
                 dist=None,
                 # direction=1
             )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            # self.test_dataset = self.instancialize(state='test')
            self.test_dataset = self.dataset(
                #  data=data,
                 data_set='test',
                 fold=self.fold,
                 batch_size=self.test_batch_size,
                 drop_last=False,
                 allow_reapt=False,
                 dist=None,
                 # direction=1
             )


    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=True)
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=0, shuffle=False)
        # dataloader = DataLoader(self.train_dataset, batch_size=None, num_workers=0, shuffle=False, persistent_workers=True)
        dataloader = DataLoader(self.train_dataset, batch_size=None, num_workers=0, shuffle=False)
        print('num_workers', dataloader.num_workers, 'persistent_workers', dataloader.persistent_workers)
        if dataloader.num_workers > 0:
            assert dataloader.persistent_workers
        return dataloader
        # return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)
        # return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=0, shuffle=False)
        # return DataLoader(self.val_dataset, batch_size=None, num_workers=0, shuffle=False, persistent_workers=True)
        return DataLoader(self.val_dataset, batch_size=None, num_workers=0, shuffle=False)

    def test_dataloader(self):
        # return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False)
        # return DataLoader(self.test_dataset, batch_size=None, num_workers=0, shuffle=False, persistent_workers=True)
        return DataLoader(self.test_dataset, batch_size=None, num_workers=0, shuffle=False)
