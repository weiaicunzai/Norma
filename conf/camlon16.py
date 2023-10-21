import os
import glob


from dataset import CAMLON16Lable

_train_dirs = {
    'wsis': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/', '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/tumor/'],
    'masks': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/', '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/tumor/']
}

_test_dirs = {
    'wsis': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/images/'],
    'masks' : ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/masks/']
}


csv_file='/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/reference.csv'

camlon16_label_fn = CAMLON16Lable(csv_file=csv_file)

# mask_path


def camlon16_wsi_filenames(data_set):
    outputs = []
    if data_set == 'train':
        for wsi_dir in _train_dirs['wsis']:
            # for  in glob.iglob(os.path.join(wsi_dir)):
            for wsi in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
                outputs.append(wsi)

        return outputs

    if  data_set == 'test':
        for wsi_dir in _test_dirs['wsis']:
            # for wsi_dir in _test_dirs['wsis']:
            for wsi in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
                outputs.append(wsi)

        return outputs
