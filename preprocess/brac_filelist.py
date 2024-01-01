import os
import glob
import pandas as pd


def func(row):
    label_dict = {'IDC':0, 'ILC':1}
    return label_dict[row['name']]

path = '/data/hdd1/by/HIPT/2-Weakly-Supervised-Subtyping/dataset_csv/tcga_brca_subset.csv.zip'

data = pd.read_csv(path)
data = data.loc[data['oncotree_code'].isin(['IDC', 'ILC'])]
data = data[['slide_id', 'oncotree_code']]
data = data.reset_index(drop=True)
data = data.rename({'oncotree_code':'name'}, axis='columns')
data['label'] = data.apply(func, axis=1)

data.to_csv('/data/hdd1/by/tmp_folder/dataset/dataset_csv/brac/brac.csv', index=False)
