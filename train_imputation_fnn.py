import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from deepctr.feature_column import SparseFeat,get_feature_names
from deepctr.models import FNN
import pandas as pd

ds_root = 'data'
dataset = 'ml-100k'
fold = 1

train_df = pd.read_csv('{}/{}/train_df_{}.csv'.format(ds_root, dataset, fold))
test_df = pd.read_csv('{}/{}/test_df_{}.csv'.format(ds_root, dataset, fold))
all_df = pd.concat([train_df, test_df])

# params
rank = 70
batch_size = 256
epochs = 100

param = {
    "dnn_hidden_units": (16, 16),
    "l2_reg_linear": 0.03,
    "l2_reg_embedding": 0.03,
    "l2_reg_dnn": 0,
    "seed": 1024,
    "dnn_dropout": 0.03,
    "dnn_activation": 'elu',
    # "dnn_use_bn": False
}

# Data
ffm_train_df = train_df.copy()
ffm_test_df = test_df.copy()

# fit
sparse_features = ["user", "item"]
target = ['rating']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    ffm_train_df[feat] = lbe.fit_transform(ffm_train_df[feat])
    ffm_test_df[feat] = lbe.fit_transform(ffm_test_df[feat])
    all_df[feat] = lbe.fit_transform(all_df[feat])

# 2.count #unique features for each sparse field
fixlen_feature_columns_train = [SparseFeat(feat, ffm_train_df[feat].nunique(),embedding_dim=rank)
                                for feat in sparse_features]
fixlen_feature_columns_test = [SparseFeat(feat, ffm_test_df[feat].nunique(),embedding_dim=rank)
                                for feat in sparse_features]

fixlen_feature_columns_all = [SparseFeat(feat, all_df[feat].nunique(),embedding_dim=rank)
                                for feat in sparse_features]

linear_feature_columns_train = fixlen_feature_columns_train
linear_feature_columns_test = fixlen_feature_columns_test
linear_feature_columns_all = fixlen_feature_columns_all

dnn_feature_columns_train = fixlen_feature_columns_train
dnn_feature_columns_test = fixlen_feature_columns_test
dnn_feature_columns_all = fixlen_feature_columns_all

# feature_names = get_feature_names(linear_feature_columns_train + dnn_feature_columns_train)
feature_names = get_feature_names(linear_feature_columns_all + dnn_feature_columns_all)

# 3.generate input data for model
train_model_input = {name:ffm_train_df[name].values for name in feature_names}
test_model_input = {name:ffm_test_df[name].values for name in feature_names}

# fit
impu_model = FNN(linear_feature_columns_all, dnn_feature_columns_all, task="regression", **param)
optimizer = tf.optimizers.RMSprop(learning_rate=0.0001)
impu_model.compile('adam', "mse", metrics=['mse', tf.keras.metrics.RootMeanSquaredError(name='RMSE')], )

imp_type = 'FNN'
impu_checkpoint_filepath = '{}/{}/ImputationModel_{}_checkpoint_{}'.format(ds_root, dataset, imp_type, fold)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=impu_checkpoint_filepath,
    save_weights_only=True,
    monitor='val_RMSE',
    mode='min',
    save_best_only=True)

history = impu_model.fit(train_model_input, ffm_train_df[target].values,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        # validation_split=0.1,
                        validation_data=(test_model_input, ffm_test_df[target].values),
                        callbacks=[model_checkpoint_callback]
                        )

# compute RMSE
pred_ans = impu_model.predict(test_model_input, batch_size=batch_size)
rmse = np.sqrt(mean_squared_error(ffm_test_df[target].values, pred_ans))

print("Test RMSE: {}" .format(rmse))