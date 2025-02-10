import tensorflow as tf
import numpy as np

class MF(tf.keras.Model):
    def __init__(self, n_users, n_items, reg_factor=1e-4, reg_bias=1e-4, mu=0.5, emb_dim=100, learning_rate=1e-3, name='MF'):
        super(MF, self).__init__(name=name)
        initializer = tf.keras.initializers.GlorotNormal(seed=2023)

        self.emb_dim = emb_dim

        self.user_emb = tf.keras.layers.Embedding(n_users, emb_dim, embeddings_initializer=initializer, name="user_embedding")
        self.item_emb = tf.keras.layers.Embedding(n_items, emb_dim, embeddings_initializer=initializer, name="item_embedding")

        self.user_bias = tf.Variable(tf.random.uniform((n_users, ), 0, 1, seed=2023), trainable=True, name="user_bias")
        self.item_bias = tf.Variable(tf.random.uniform((n_items, ), 0, 1, seed=2023), trainable=True, name="item_bias")

        self.mu = tf.constant(mu, dtype=tf.float32)
        self.reg_factor = reg_factor
        self.reg_bias = reg_bias
        self.n_users = n_users
        self.n_items = n_items

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, data):
        users, items, ratings = data
        ratings = tf.cast(ratings, tf.float32)

        predictions = self.mu + tf.gather(self.user_bias, users) + tf.gather(self.item_bias, items) + tf.reduce_sum(self.user_emb(users) * self.item_emb(items), axis=-1)
        return predictions

    def train_step(self, data):
        users, items, ratings = data
        ratings = tf.cast(ratings, tf.float32)

        with tf.GradientTape() as tape:
            predictions = self.mu + tf.gather(self.user_bias, users) + tf.gather(self.item_bias, items) + tf.reduce_sum(self.user_emb(users) * self.item_emb(items), axis=-1)
            loss = tf.reduce_sum(tf.math.square(ratings - predictions))
            loss += tf.reduce_sum(self.reg_factor * tf.math.square(self.user_emb(users)))
            loss += tf.reduce_sum(self.reg_factor * tf.math.square(self.item_emb(items)))
            loss += tf.reduce_sum(self.reg_bias * tf.math.square(tf.gather(self.user_bias, users)))
            loss += tf.reduce_sum(self.reg_bias * tf.math.square(tf.gather(self.item_bias, items)))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {'loss': loss}

    def test_step(self, data):
        users, items, ratings = data
        ratings = tf.cast(ratings, tf.float32)

        predictions = self.mu + tf.gather(self.user_bias, users) + tf.gather(self.item_bias, items) + tf.reduce_sum(self.user_emb(users) * self.item_emb(items), axis=-1)

        mse = tf.keras.metrics.MSE(ratings, predictions)
        rmse = tf.math.sqrt(mse)
        # rmse_func = tf.keras.metrics.RootMeanSquaredError()
        # rmse = rmse_func(ratings, predictions)

        return {'mse': mse, 'rmse': rmse}

    def get_predictions(self):
        # users = tf.keras.backend.arange(self.n_users)
        # users = tf.tile(tf.expand_dims(users, 0), (self.n_items, 1))
        # users = tf.reshape(users, (self.n_items * self.n_users, ))

        users = tf.keras.backend.arange(self.n_users)
        users = tf.reshape(users, (-1, 1))
        users = tf.tile(users, (1, self.n_items))
        users = tf.reshape(users, (self.n_items * self.n_users, ))

        items = tf.keras.backend.arange(self.n_items)
        items = tf.tile(tf.expand_dims(items, 0), (1, self.n_users))
        items = tf.reshape(items, tf.shape(users))

        predictions = self.mu + tf.gather(self.user_bias, users) + tf.gather(self.item_bias, items) + tf.reduce_sum(self.user_emb(users) * self.item_emb(items), axis=-1)
        predictions = tf.reshape(predictions, (self.n_users, self.n_items))

        return predictions

    def get_predictions_by_emb(self, ue, ie):
        users = tf.keras.backend.arange(self.n_users)
        users = tf.reshape(users, (-1, 1))
        users = tf.tile(users, (1, self.n_items))
        users = tf.reshape(users, (self.n_items * self.n_users, ))

        items = tf.keras.backend.arange(self.n_items)
        items = tf.tile(tf.expand_dims(items, 0), (1, self.n_users))
        items = tf.reshape(items, tf.shape(users))

        predictions = self.mu + tf.gather(self.user_bias, users) + tf.gather(self.item_bias, items) + tf.reduce_sum(tf.gather(ue, users) * tf.gather(ie, items), axis=-1)
        predictions = tf.reshape(predictions, (self.n_users, self.n_items))

        return predictions