import numpy as np
import statistics


def get_metric(ground_truth, predictions):

    # MSE, RMSE
    ground_truth_mask = (ground_truth > 0).astype(np.float32)
    num_ground_truth = ground_truth_mask.sum()
    mse = np.square(ground_truth - predictions * ground_truth_mask).sum() / num_ground_truth
    rmse = np.sqrt(mse)

    # User Fairness
    user_mse = np.square(ground_truth - predictions * ground_truth_mask).sum(axis=-1) / (ground_truth_mask.sum(axis=-1) + 1e-9)
    user_mse = user_mse[ground_truth_mask.sum(axis=-1) > 0]
    user_unfairness = statistics.variance(user_mse)

    # item fairness
    item_unfairness = np.square(predictions - np.reshape(predictions.mean(axis=-1), (-1, 1))).mean()

    return mse, rmse, user_unfairness, item_unfairness