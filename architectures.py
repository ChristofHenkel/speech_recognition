import tensorflow as tf
from tensorflow.contrib import layers


class Model1:

    def __init__(self, cfg):
        self.test = ''
        self.config = cfg



    @staticmethod
    def calc_logits(x,keep_prob,is_training,use_batch_norm,num_classes):


        x2 = layers.batch_norm(x, is_training=is_training)

        x2 = layers.conv2d(x2, 16, 3, 1,
                           activation_fn=tf.nn.elu,
                           normalizer_fn=layers.batch_norm if use_batch_norm else None,
                           normalizer_params={'is_training': is_training}
                           )

        x2 = layers.max_pool2d(x2, 2, 2)

        x2 = layers.conv2d(x2, 32, 3, 1,
                           activation_fn=tf.nn.elu,
                           normalizer_fn=layers.batch_norm if use_batch_norm else None,
                           normalizer_params={'is_training': is_training}
                           )

        x2 = layers.max_pool2d(x2, 2, 2)

        mpool = tf.reduce_max(x2, axis=[1, 2], keep_dims=True)
        apool = tf.reduce_mean(x2, axis=[1, 2], keep_dims=True)

        x2 = 0.5 * (mpool + apool)
        # we can use conv2d 1x1 instead of dense

        # (128, 1, 1, 32) -> (128, 1, 1, 32)
        x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)

        # again conv2d 1x1 instead of dense layer
        # (128, 1, 1, 32) -> (128, 1, 1, 12)
        # x2 = layers.conv2d(x2, num_classes, 1, 1, activation_fn=None)
        x2 = layers.fully_connected(x2, num_classes, activation_fn=tf.nn.relu)

        # -> (128, 1, 1, 12) - > (128, 12)
        logits = tf.squeeze(x2, [1, 2])
        return logits