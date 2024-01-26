




import pandas as pd

file = pd.read_excel('/data/hdd2/by/tmp_folder/BRACS.xlsx')
# print(file)
file['WSI Filename'] = file['WSI Filename'].apply(lambda x: x + '.svs')
file.drop('Patient Id', axis=1, inplace=True)
file.drop('RoI ', axis=1, inplace=True)
file.drop('Set', axis=1, inplace=True)
file.rename(columns={'WSI Filename': 'slide_id'}, inplace=True)
file.rename(columns={'WSI label': 'name'}, inplace=True)

mapping = {}
for k, v in zip(file['name'].unique(), range(7)):
    mapping[k] = v


file['label'] = file['name'].apply(lambda x: mapping[x])






file.to_csv('/data/hdd2/by/tmp_folder/datasets/dataset_csv/bracs/bracs.csv', index=False)
