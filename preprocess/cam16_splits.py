import pandas as pd
import os


csv_path = 'dataset/dataset_csv/cam16/cam16.csv'

df = pd.read_csv(csv_path)

def removeext(slide_id):
    path = os.path.splitext(slide_id)[0]
    print(path)
    return path

df['slide_id'] = df['slide_id'].apply(removeext)
# print(df)

# print(data['slide_id'] == )
# print(df.loc['test' in df['slide_id'] ])
# print(df.loc(df['column_name'].isin(some_values)))
# print(df['slide_id'].isin('test'))
# print(df['test' in df['slide_id']])
# print()
test = df[df['slide_id'].str.contains('test')]
train_tmp = df[df['slide_id'].str.contains('normal') | df['slide_id'].str.contains('tumor')]
# print(train.shape)
train = train_tmp.sample(frac=0.9, random_state=42)
# print(train.shape)
# import sys; sys.exit()
val = train_tmp.drop(train.index)
# print(df['slide_id'][val_idx])
# print(df['slide_id'].isin(val['slide_id']))

csv_bool = pd.concat([
    df['slide_id'],
    df['slide_id'].isin(train['slide_id']),
    df['slide_id'].isin(val['slide_id']),
    df['slide_id'].isin(test['slide_id']),
], axis=1)

# print(csv_bool)
# print(csv_bool.columns)
csv_bool.columns = ['', 'train', 'val', 'test']

# print(csv_bool.columns)
# print(csv_bool)
# csv_bool.set_index('slide_id', inplace=True)
csv_bool.to_csv('dataset/splits/cam16/splits_{}_bool.csv'.format(0), index=False)
# print(csv_bool.index)

train['slide_id'].reset_index(drop=True, inplace=True),
val['slide_id'].reset_index(drop=True, inplace=True),
test['slide_id'].reset_index(drop=True, inplace=True)
splits =  pd.concat(
    [
        train['slide_id'],
        val['slide_id'],
        test['slide_id']
    ],
    axis=1
)
splits.columns = ['train', 'val', 'test']
print(splits)

# splits.to_csv('')
splits.to_csv('dataset/splits/cam16/splits_{}.csv'.format(0))