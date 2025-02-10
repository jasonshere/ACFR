import tensorflow as tf
import pandas as pd
from base_model import MF

ds_root = 'data'
dataset = 'ml-100k'
fold = 1

train_df = pd.read_csv('{}/{}/train_df_{}.csv'.format(ds_root, dataset, fold))
test_df = pd.read_csv('{}/{}/test_df_{}.csv'.format(ds_root, dataset, fold))
all_df = pd.concat([train_df, test_df])

n_users = all_df.user.nunique()
n_items = all_df.item.nunique()


batch_size = 256

train_ds = tf.data.Dataset.from_tensor_slices((train_df.user.values, train_df.item.values, train_df.rating.values))
test_ds = tf.data.Dataset.from_tensor_slices((test_df.user.values, test_df.item.values, test_df.rating.values))

mf_model = MF(n_users, n_items, mu=train_df.rating.mean(), reg_factor=1e-3, reg_bias=1e-3, emb_dim=10, learning_rate=1e-4)
mf_model.compile(optimizer='adam', loss=None, run_eagerly=False, weighted_metrics=[])
mf_model([1, 1, 1], training=False)
# mf_model.load_weights('{}/{}/MF_{}.h5'.format(ds_root, dataset, fold))

checkpoint_filepath = '{}/{}/MF_checkpoint_{}'.format(ds_root, dataset, fold)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_rmse',
    mode='min',
    save_best_only=True)

mf_model.fit(train_ds.batch(batch_size), epochs=25, validation_freq=1, validation_data=test_ds.batch(batch_size), callbacks=[model_checkpoint_callback])