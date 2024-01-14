
# import os
# import sys
# sys.path.append(os.getcwd())
# import cv2
# from utils.vis import vis_mask














# if __name__ == '__main__':

#     wsi_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/normal_045.tif'
#     # mask_path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/normal_045.png'
#     mask_path = 'tmp1/mask.png'

#     out = vis_mask(wsi_path, mask_path, 6)

#     cv2.imwrite('test.jpg', out)

# import os
# import glob
import pandas as pd
import csv
import os

csv_file = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/reference.csv'
def func(name):
    label_dict = {'Normal':0, 'Tumor':1}
    return label_dict[name]

# path = '/data/hdd1/by/HIPT/2-Weakly-Supervised-Subtyping/dataset_csv/tcga_brca_subset.csv.zip'

# data = pd.read_csv(csv_file)
# print(data)
test_file_lists = {}
with open(csv_file) as f:
    for row in csv.reader(f):
        # print(row)
        print(row)
        test_file_lists[row[0]] = row[1]

# print(file_lists)
# file_lists[]
'slide_id,name,label'

all_list = {
    'slide_id': [],
    'name': [],
    'label': []
}

for i in os.listdir('/data/smb/syh/WSI_cls/cam16/img'):

    # https://rumc-gcorg-p-public.s3.amazonaws.com/f/challenge/80/105788c6-176a-4dc3-89cf-62f4f37d1484/camelyon16_readme.md
    # https://github.com/hrzhang1123/DTFD-MIL/issues/7
    if 'normal_86' in i:
        raise ValueError('Originally misclassified, renamed to tumor_111')

    if 'test_049' in i:
        raise ValueError('Duplicate slide')

    all_list['slide_id'].append(i)
    if 'tumor' in i:
        all_list['name'].append('Tumor')
    elif 'normal' in i:
        all_list['name'].append('Normal')
    else:
        print(i)
        name = test_file_lists[i.split('.')[0]]
        all_list['name'].append(name)


    name = all_list['name'][-1]
    all_list['label'].append(func(name))

# print(all_list)
    # print(i)

df = pd.DataFrame(all_list)
print(df)

df.to_csv('dataset/dataset_csv/cam16/cam16.csv',  index=False)

# data = data.loc[data['oncotree_code'].isin(['IDC', 'ILC'])]
# data = data[['slide_id', 'oncotree_code']]
# data = data.reset_index(drop=True)
# data = data.rename({'oncotree_code':'name'}, axis='columns')
# data['label'] = data.apply(func, axis=1)

# data.to_csv('/data/hdd1/by/tmp_folder/dataset/dataset_csv/brac/brac.csv', index=False)
