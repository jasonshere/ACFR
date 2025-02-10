import numpy as np
import lenskit.crossfold as xf
import os
from data.data_loader import ML100KDatasetLoader

ds_name = 'ml-100k'
ds_loader = ML100KDatasetLoader(ds_name=ds_name)
ratings = ds_loader.ratings()
ratings.user_id = ratings.user_id - 1
ratings.item_id = ratings.item_id - 1

original_item_id = ratings.item_id.unique().tolist()
new_item_id = np.arange(len(original_item_id)).tolist()

ratings['item_id'] = ratings['item_id'].map(dict(zip(original_item_id, new_item_id)))

original_user_id = ratings.user_id.unique().tolist()
new_user_id = np.arange(len(original_user_id)).tolist()

ratings['user_id'] = ratings['user_id'].map(dict(zip(original_user_id, new_user_id)))

ds_name = 'ml-100k'
path = "data/{}" .format(ds_name)

os.makedirs(path, exist_ok=True)

ratings = ratings.rename(columns={'user_id': 'user', 'item_id': 'item'})
FOLDS = 5
N = 0.2

for i, tp in enumerate(xf.partition_rows(ratings, FOLDS)):
    print(tp.train, tp.test)
    tp.train.to_csv(r'data/{}/train_df_{}.csv'.format(ds_name, i+1), index=False)
    tp.test.to_csv(r'data/{}/test_df_{}.csv'.format(ds_name, i+1), index=False)