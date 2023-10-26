import os
import glob


from dataset.utils import CAMLON16Label

train_dirs = {
    'wsis': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/', '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/tumor/'],
    'jsons': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_json/normal/patch_size_512_at_mag_5', '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_json/tumor/patch_size_512_at_mag_5/']
}

test_dirs = {
    'wsis': ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/images/'],
    'jsons' : ['/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/jsons/patch_size_512_at_mag_5/']
}


_csv_file='/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/reference.csv'

camlon16_label_fn = CAMLON16Label(csv_file=_csv_file)


# def camlon16_wsi_filenames(data_set):
#     outputs = []
#     if data_set == 'train':
#         dirs = _train_dirs
#     else:
#         dirs = _test_dirs


#     for wsi_dir, json_dir in zip(dirs['wsis'], dirs['jsons']):
#         for wsi_path in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=true):
#             basename = os.path.basename(wsi_path)
#             json_path = os.path.join(json_dir, basename.replace('.tif', '.json'))
#             outputs.append({
#                 'wsi': wsi_path,
#                 'json': json_path
#             })

#     return outputs


    # if  data_set == 'test':
    #     wsi_dirs = _train_dirs['wsis']
    #     for wsi_dir in _test_dirs['wsis']:
    #         # for wsi_dir in _test_dirs['wsis']:
    #         for wsi in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
    #             outputs.append(wsi)

    # return outputs



class Settings:
    def __init__(self):
        self.label_fn = camlon16_label_fn
        self.train_dirs = train_dirs
        self.test_dirs = test_dirs


settings = Settings()