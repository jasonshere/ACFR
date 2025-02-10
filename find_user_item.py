import tensorflow as tf


class FindGreat(tf.keras.Model):
    def __init__(self, n_users, n_items, selected_index, ground_truth, imputed_ratings, mf_model, learning_rate=1e-3, lambda_=0.7, fair_mode='user', impu_direction='user'):
        super(FindGreat, self).__init__()
        self.mf_model = mf_model
        self.imputed_ratings = imputed_ratings
        self.selected_index = selected_index
        self.ground_truth = ground_truth
        self.impu_direction = impu_direction

        self.n_users = n_users
        self.n_items = n_items

        self.fair_mode = fair_mode

        self.lambda_ = lambda_

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_step(self, data):

        if self.impu_direction == 'user':

            with tf.GradientTape() as tape:

                predictions = self.mf_model.get_predictions()
                predictions_user = tf.gather(predictions, self.selected_index)

                users = tf.tile([[self.selected_index]], (self.n_items, 1))
                users = tf.reshape(users, (self.n_items, ))
                items = tf.keras.backend.arange(self.n_items)

                loss = tf.reduce_sum(tf.math.square(self.imputed_ratings - predictions_user))
                loss += tf.reduce_sum(self.mf_model.reg_factor * tf.math.square(self.mf_model.user_emb(users)))
                loss += tf.reduce_sum(self.mf_model.reg_factor * tf.math.square(self.mf_model.item_emb(items)))
                loss += tf.reduce_sum(self.mf_model.reg_bias * tf.math.square(tf.gather(self.mf_model.user_bias, users)))
                loss += tf.reduce_sum(self.mf_model.reg_bias * tf.math.square(tf.gather(self.mf_model.item_bias, items)))

            grads = tape.gradient(loss, self.mf_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.mf_model.trainable_weights))

        if self.impu_direction == 'item':

            with tf.GradientTape() as tape:

                predictions = self.mf_model.get_predictions()
                predictions_item = tf.gather(tf.transpose(predictions), self.selected_index)
                predictions_item = tf.reshape(predictions_item, (self.n_users, ))


                users = tf.keras.backend.arange(self.n_users)

                items = tf.tile([[self.selected_index]], (self.n_users, 1))
                items = tf.reshape(items, (self.n_users, ))

                loss = tf.reduce_sum(tf.math.square(self.imputed_ratings - predictions_item))
                loss += tf.reduce_sum(self.mf_model.reg_factor * tf.math.square(self.mf_model.user_emb(users)))
                loss += tf.reduce_sum(self.mf_model.reg_factor * tf.math.square(self.mf_model.item_emb(items)))
                loss += tf.reduce_sum(self.mf_model.reg_bias * tf.math.square(tf.gather(self.mf_model.user_bias, users)))
                loss += tf.reduce_sum(self.mf_model.reg_bias * tf.math.square(tf.gather(self.mf_model.item_bias, items)))

            grads = tape.gradient(loss, self.mf_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.mf_model.trainable_weights))

        with tf.GradientTape() as tape:
            predictions = self.mf_model.get_predictions()
            fair_loss = 0.0
            if self.fair_mode == 'user':
                ground_truth_mask = tf.cast(tf.greater(self.ground_truth, 0.0), tf.float32)
                user_mse = tf.reduce_sum(tf.math.square(self.ground_truth - predictions * ground_truth_mask), axis=-1) / (tf.reshape(tf.reduce_sum(ground_truth_mask, axis=-1), (-1, 1)) + 1e-5)
                user_unfairness = tf.reduce_mean(tf.math.square(user_mse - tf.reduce_mean(user_mse)))

                fair_loss = self.lambda_ * user_unfairness

            if self.fair_mode == 'item':
                item_unfairness = tf.reduce_mean(tf.math.square(predictions - tf.reshape(tf.reduce_mean(predictions, axis=-1), (-1, 1))))
                fair_loss = self.lambda_ * item_unfairness

        grads = tape.gradient(fair_loss, self.mf_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.mf_model.trainable_weights))


        return {}