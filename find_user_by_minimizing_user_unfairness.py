from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
import tensorflow as tf
import pandas as pd
from metrics import get_metric
from find_user_item import FindGreat
from base_model import MF
from deepctr.feature_column import SparseFeat,get_feature_names
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from deepctr.models import DeepFM

ds_root = 'data'
dataset = 'ml-100k'
fold = 1

train_df = pd.read_csv('{}/{}/train_df_{}.csv'.format(ds_root, dataset, fold))
test_df = pd.read_csv('{}/{}/test_df_{}.csv'.format(ds_root, dataset, fold))
all_df = pd.concat([train_df, test_df])

n_users = all_df.user.nunique()
n_items = all_df.item.nunique()

train_ds = tf.data.Dataset.from_tensor_slices((train_df.user.values, train_df.item.values, train_df.rating.values))
test_ds = tf.data.Dataset.from_tensor_slices((test_df.user.values, test_df.item.values, test_df.rating.values))

checkpoint_filepath = '{}/{}/MF_checkpoint_{}'.format(ds_root, dataset, fold)
mf_model = MF(n_users, n_items, mu=train_df.rating.mean(), reg_factor=1e-3, reg_bias=1e-3, emb_dim=10, learning_rate=1e-4)
mf_model.compile(optimizer='adam', loss=None, run_eagerly=False, weighted_metrics=[])
mf_model([1, 1, 1], training=False)

selected_score = []
lambda_ = 0.0001
fair_obj = 'user'
fair_lambda_ = 1.0
imp_type = 'deepfm' # deepfm, average, random
impu_direction = 'user' # user, item
imputed_ratings = None

if impu_direction == 'user':
    # set_index = range(n_users)
    # Create a random number generator
    rng = default_rng()
    # Generate 10 unique random numbers between 0 and 19 (inclusive)
    set_index = rng.choice(n_users, size=150, replace=False)


if impu_direction == 'item':
    # set_index = range(n_users)
    # Create a random number generator
    rng = default_rng()
    # Generate 10 unique random numbers between 0 and 19 (inclusive)
    set_index = rng.choice(n_items, size=150, replace=False)

for selected_index in tqdm(set_index):

    mf_model.load_weights(checkpoint_filepath)
    predictions = mf_model.get_predictions().numpy()
    train_ground_truth = np.zeros((n_users, n_items))
    train_ground_truth[train_df['user'].values, train_df['item'].values] = train_df['rating'].values
    mse, rmse, user_unfairness, item_unfairness = get_metric(train_ground_truth, predictions)

    # imputed_ratings
    ground_truth_mask = (train_ground_truth > 0).astype(np.float32)

    if imp_type == 'deepfm':

        # params
        rank = 10
        batch_size = 256
        epochs = 15

        param = {
            "dnn_hidden_units": (20, 20, ),
            "l2_reg_linear": 0.001,
            "l2_reg_embedding": 0.001,
            "l2_reg_dnn": 0.001,
            "seed": 2023,
            "dnn_dropout": 0.001,
            "dnn_activation": 'relu',
            "dnn_use_bn": False
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
        impu_model = DeepFM(linear_feature_columns_all, dnn_feature_columns_all, task="regression", **param)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        impu_model.compile(optimizer='adam', loss="mse", metrics=['mse', tf.keras.metrics.RootMeanSquaredError(name='RMSE')], )

        impu_checkpoint_filepath = '{}/{}/ImputationModel_{}_checkpoint_{}'.format(ds_root, dataset, imp_type, fold)
        status = impu_model.load_weights(impu_checkpoint_filepath)
        status.expect_partial()

        if impu_direction == 'user':
            imputed_predictions = impu_model.predict({'user': np.array([selected_index] * n_items), 'item': np.arange(n_items)})
            imputed_predictions = np.reshape(imputed_predictions, (n_items, ))
            imputed_ratings = ground_truth_mask[selected_index] * train_ground_truth[selected_index] + (1 - ground_truth_mask[selected_index]) * imputed_predictions

        if impu_direction == 'item':
            imputed_predictions = impu_model.predict({'user': np.arange(n_users), 'item': np.array([selected_index] * n_users)})
            imputed_predictions = np.reshape(imputed_predictions, (n_users, ))
            imputed_ratings = ground_truth_mask[:, selected_index] * train_ground_truth[:, selected_index] + (1 - ground_truth_mask[:, selected_index]) * imputed_predictions

    model = FindGreat(n_users=n_users, n_items=n_items, selected_index=selected_index, ground_truth=train_ground_truth, fair_mode=fair_obj, imputed_ratings=imputed_ratings, mf_model=mf_model, learning_rate=1e-4, lambda_=fair_lambda_, impu_direction=impu_direction)
    model.compile(optimizer='adam', loss=None, run_eagerly=False, weighted_metrics=['rmse', 'mse'])
    model.fit(train_ds.take(1), epochs=10, verbose=0)

    if impu_direction == 'user':
        # predict
        predictions = mf_model.get_predictions().numpy()
        predictions_user = predictions[selected_index]

        mse_after, rmse_after, user_unfairness_after, item_unfairness_after = get_metric(train_ground_truth, predictions)
        change = tf.keras.metrics.MSE(predictions_user, imputed_ratings).numpy()

    if impu_direction == 'item':
        # predict
        predictions = mf_model.get_predictions().numpy()
        predictions_item = predictions[:, selected_index]
        predictions_item = np.reshape(predictions_item, (n_users, ))

        mse_after, rmse_after, user_unfairness_after, item_unfairness_after = get_metric(train_ground_truth, predictions)
        change = tf.keras.metrics.MSE(predictions_item, imputed_ratings).numpy()

    score = 0.0
    if fair_obj == 'user':
        score = (user_unfairness - user_unfairness_after) - lambda_ * change

    if fair_obj == 'item':
        score = (item_unfairness - item_unfairness_after) - lambda_ * change

    selected_score.append(score)

    print("Object: {}, Score: {}, Ori Fairness: {}, Fairness: {}, Change: {}".format(selected_index, score, user_unfairness, user_unfairness_after, change))

selected_score = np.array(selected_score)
pd.DataFrame(selected_score).to_csv('{}/{}/{}_score_{}_{}_{}.csv'.format(ds_root, dataset, fair_obj, imp_type, impu_direction, fold), index=False)