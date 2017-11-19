from batch_gen import BatchGen
import tensorflow as tf
from tensorflow.contrib import layers, signal
import numpy as np

class Config:
    soundcorpus_fp = 'assets/corpora/corpus1/validation.soundcorpus.part1.p'
    batch_size = 1000
    size = 16000
    is_training = True
    use_batch_norm = True
    keep_prob = 0.9

cfg = Config()
gen = BatchGen(batch_size = cfg.batch_size,soundcorpus_fp = cfg.soundcorpus_fp)
decoder = gen.decoder
num_classes=len(decoder)

# x,y = next(gen.batch_gen())


# Define Graph

size = cfg.size
batch_size = cfg.batch_size
is_training = cfg.is_training

graph = tf.Graph()
with graph.as_default():
    # tf Graph input
    with tf.name_scope("Input"):

        x = tf.placeholder(tf.float32, shape=(None, size), name="input")
        x.set_shape([batch_size, size])
        y = tf.placeholder(tf.int64, shape=(None, 1), name="input")
        # keep_prob = tf.placeholder(tf.float32, name="dropout")

    specgram = signal.stft(
        x,
        400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        160,  # 16000 * 0.010 -- default stride
    )
    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))

    #x = tf.stack([amp, phase], axis=3)  # shape is [bs, time, freq_bins, 2]
    #x = tf.to_float(x)


    #x = layers.batch_norm(x, is_training=is_training)
    #for i in range(4):
    #    x = layers.conv2d(
    #        x, 16 * (2 ** i), 3, 1,
    #        activation_fn=tf.nn.elu,
    #        normalizer_fn=layers.batch_norm if cfg.use_batch_norm else None,
    #        normalizer_params={'is_training': is_training}
    #    )
    #    x = layers.max_pool2d(x, 2, 2)

    ## just take two kind of pooling and then mix them, why not :)
    #mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
    #apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    #x = 0.5 * (mpool + apool)
    # we can use conv2d 1x1 instead of dense
    #x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
    #x = tf.nn.dropout(x, keep_prob=cfg.keep_prob if is_training else 1.0)

    # again conv2d 1x1 instead of dense layer
    # logits = layers.conv2d(x, num_classes, 1, 1, activation_fn=None)
    logits = specgram

# Launch the graph
# TESTING
def debug_model():
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_x, batch_y = next(gen.batch_gen())
        s,p,a = sess.run([phase, amp, specgram], feed_dict={x: batch_x, y: batch_y})
        # l = sess.run([logits], feed_dict={x: batch_x, y: batch_y, keep_prob: keep_probability})
        return s,p,a

s,p,a = debug_model()
