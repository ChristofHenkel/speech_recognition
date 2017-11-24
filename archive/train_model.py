"""
# Good MFCC explanation:
http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""

from batch_gen import BatchGen
import tensorflow as tf
from tensorflow.contrib import layers, signal
import numpy as np
import time
import logging

logging.basicConfig(level=logging.DEBUG)

class Config:
    soundcorpus_fp = 'assets/corpora/corpus3/trainp.soundcorpus.p'
    batch_size = 128
    size = 16000
    is_training = True
    use_batch_norm = True
    keep_prob = 0.9
    max_gradient = 5
    learning_rate = 1
    training_iters = 52200
    display_step = 10
    epochs = 2

cfg = Config()
gen = BatchGen(batch_size = cfg.batch_size,soundcorpus_fp = cfg.soundcorpus_fp)
decoder = gen.decoder
num_classes=len(decoder)

# Define Graph

size = cfg.size
batch_size = cfg.batch_size
is_training = cfg.is_training
max_gradient = cfg.max_gradient

def preprocess(x):
    specgram = signal.stft(
        x,
        400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        160,  # 16000 * 0.010 -- default stride

    )
    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))
    x2 = tf.stack([amp, phase], axis=3)  # shape is [bs, time, freq_bins, 2]
    x2 = tf.to_float(x2)

    # # Compute MFCC using Tensorflow functions
    # # A 400-point STFT with frames of 25 ms and 10 ms overlap.
    # sample_rate = 16000
    # stfts = tf.contrib.signal.stft(x, frame_length=400, frame_step=160,
    #                                fft_length=400)
    # spectrograms = tf.abs(stfts)
    #
    # # Warp the linear scale spectrograms into the mel-scale.
    # num_spectrogram_bins = stfts.shape[-1].value
    # lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    # linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
    #   num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    #   upper_edge_hertz)
    # mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    # mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    #   linear_to_mel_weight_matrix.shape[-1:]))
    #
    # # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    # log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
    #
    # # Compute MFCCs from log_mel_spectrograms and take the first 13.
    # mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
    #   log_mel_spectrograms)[..., :13]
    # mfccs = tf.Print(mfccs, [mfccs], message="MFCCs: ")
    # delta_mfccs = np.append(mfccs[0], mfccs[1:] - mfccs[:-1])
    # dd_mfccs = np.append(delta_mfccs[0], delta_mfccs[1:] - delta_mfccs[:-1])
    # x2 = tf.stack([mfccs, delta_mfccs, dd_mfccs], axis=3)  # shape is [bs, time, freq_bins, ???]

    return x2


graph = tf.Graph()
with graph.as_default():
    # tf Graph input
    with tf.name_scope("Input"):

        x = tf.placeholder(tf.float32, shape=(None, size), name="input")
        # x.set_shape([batch_size, size])
        y = tf.placeholder(tf.int64, shape=(None, ), name="input")
        keep_prob = tf.placeholder(tf.float32, name="dropout")

    x2 = preprocess(x)

    x2 = layers.batch_norm(x2, is_training=is_training)



    # Convlayer1 input dim (128, 98, 257, 2), output dim (128, 98, 257, 16)
    x2 = layers.conv2d(x2, 16, 3, 1,
                       activation_fn=tf.nn.elu,
                       normalizer_fn=layers.batch_norm if cfg.use_batch_norm else None,
                       normalizer_params={'is_training': is_training}
                       )
    # (128, 98, 257, 16) -> (128, 49, 128, 16)
    x2 = layers.max_pool2d(x2, 2, 2)

    # Convlay2 (128, 49, 128, 16) -> (128, 49, 128, 32)
    x2 = layers.conv2d(x2, 32, 3, 1,
                       activation_fn=tf.nn.elu,
                       normalizer_fn=layers.batch_norm if cfg.use_batch_norm else None,
                       normalizer_params={'is_training': is_training}
                       )
    # (128, 49, 128, 32) -> (128, 24, 64, 32)
    x2 = layers.max_pool2d(x2, 2, 2)

    # Convlay3  (128, 24, 64, 32) -> (128, 24, 64, 64)
    x2 = layers.conv2d(x2, 64, 3, 1,
                       activation_fn=tf.nn.elu,
                       normalizer_fn=layers.batch_norm if cfg.use_batch_norm else None,
                       normalizer_params={'is_training': is_training}
                       )
    # (128, 24, 64, 64) -> (128, 12, 32, 64)
    x2 = layers.max_pool2d(x2, 2, 2)

    # Convlay4 (128, 12, 32, 64) -> (128, 12, 32, 128)
    x2 = layers.conv2d(x2, 128, 3, 1,
                       activation_fn=tf.nn.elu,
                       normalizer_fn=layers.batch_norm if cfg.use_batch_norm else None,
                       normalizer_params={'is_training': is_training}
                       )
    # (128, 12, 32, 128) -> (128,6,16,128)
    x2 = layers.max_pool2d(x2, 2, 2)

    ## just take two kind of pooling and then mix them, why not :)

    # (128,6,16,128) -> (128, 1, 1, 128)
    mpool = tf.reduce_max(x2, axis=[1, 2], keep_dims=True)

    # (128,6,16,128) -> (128, 1, 1, 128)
    apool = tf.reduce_mean(x2, axis=[1, 2], keep_dims=True)

    # (128, 1, 1, 128)
    x2 = 0.5 * (mpool + apool)
    # we can use conv2d 1x1 instead of dense

    # (128, 1, 1, 128) -> (128, 1, 1, 128)
    x2 = layers.conv2d(x2, 128, 1, 1, activation_fn=tf.nn.elu)
    x2 = tf.nn.dropout(x2, keep_prob=keep_prob)

    # again conv2d 1x1 instead of dense layer
    # (128, 1, 1, 128) -> (128, 1, 1, 12)
    # x2 = layers.conv2d(x2, num_classes, 1, 1, activation_fn=None)
    x2 = layers.fully_connected(x2,num_classes,activation_fn=tf.nn.relu)

    # -> (128, 1, 1, 12) - > (128, 12)
    logits = tf.squeeze(x2, [1, 2])

    # (128, 12) -> (1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))


    gradients, _ = tf.clip_by_global_norm(tf.gradients(loss, tf.trainable_variables()),
                                          max_gradient, name="clip_gradients")
    iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)
    #if cfg.lr_decay != None:
    #    learning_rate = tf.train.exponential_decay(cfg.learning_rate, iteration,
    #                                               100000, cfg.lr_decay, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=cfg.learning_rate).apply_gradients(
        zip(gradients, tf.trainable_variables()),
        name="train_step",
        global_step=iteration)

    #probs = tf.nn.softmax(logits2)
    pred = tf.argmax(logits, axis=-1)
    #pred = tf.argmax(logits2, 1)
    correct_pred = tf.equal(pred, tf.reshape(y, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #pred = tf.argmax(logits2, axis=-1)
    #accuracy, acc_op = tf.metrics.mean_per_class_accuracy(y, pred, num_classes)

# Launch the graph
# TESTING


def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

def debug_model():
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_x, batch_y = next(gen.batch_gen())
        _x3, l, acc = sess.run([logits, loss,accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})
        print(_x3.shape, l, acc)
        return l, acc



def train_model():
    with tf.Session(graph=graph) as sess:
        logging.info('Start training')
        init = tf.global_variables_initializer()
        # summary_writer = tf.summary.FileWriter(cfg.logs_path, graph=graph)
        sess.run(init)
        for epoch in range(cfg.epochs):
            step = 1
            # Todo
            #redefine generator to start from beginning of corpus
            # gen =

            # Keep training until reach max iterations
            current_time = time.time()
            while step * batch_size < cfg.training_iters:
                logging.info('step ' + str(step))
                batch_x, batch_y = next(gen.batch_gen())

                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})
                if step % cfg.display_step == 0:
                    # Calculate batch accuracy

                    logging.info('runtime for batch of ' + str(cfg.batch_size * cfg.display_step) + ' ' + str(time.time()-current_time))
                    current_time = time.time()
                    l, acc= sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})

                    print(l, acc)


                step += 1

        print("Optimization Finished!")

if __name__ == '__main__':
    # l, acc = debug_model()
    train_model()

