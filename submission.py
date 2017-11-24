import tensorflow as tf
import numpy as np
from glob import glob
from batch_gen import SoundCorpus
from architectures import Model1
import os
import pickle
import logging

class Config:
    soundcorpus_dir = 'assets/corpora/corpus7/'
    batch_size = 6
    is_training = False
    use_batch_norm = True
    keep_prob = 0.8
    max_gradient = 5
    learning_rate = 0.5
    display_step = 10
    epochs = 10
    logs_path = 'models/model5/logs/'

cfg = Config()

corpus = SoundCorpus(cfg.soundcorpus_dir,mode='test')

batch_gen = corpus.batch_gen(cfg.batch_size)

decoder = corpus.decoder
num_classes=len(decoder)


# set_graph Graph

batch_size = cfg.batch_size
is_training = cfg.is_training
max_gradient = cfg.max_gradient

training_iters = corpus.len

graph = tf.Graph()
with graph.as_default():
    # tf Graph input
    tf.set_random_seed(3)
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, shape=(None, 99, 13, 3), name="input")
        # x.set_shape([batch_size, size])
        y = tf.placeholder(tf.int64, shape=(None,), name="input")
        keep_prob = tf.placeholder(tf.float32, name="dropout")


    with tf.variable_scope('logit'):
        logits = Model1.calc_logits(x, keep_prob, is_training, cfg.use_batch_norm, num_classes)
        predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        cost = tf.reduce_mean(xent, name='xent')
        # cost += self._decay()

        tf.summary.scalar('cost', cost)

    with tf.variable_scope('acc'):
        pred = tf.argmax(logits, 1)
        correct_prediction = tf.equal(pred, tf.reshape(y, [-1]))
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name='accu')

        tf.summary.scalar('accuracy', accuracy)
    saver = tf.train.Saver()

def submission():
    fn_model = 'models/model6/logs/model_mfcc_bsize256_e0.ckpt'
    # %%
    id2name = corpus.decoder
    cfg = Config()
    # cfg.soundcorpus_fp = 'assets/corpora/corpus7/test.pm.soundcorpus.p'
    size = 158538

    with tf.Session(graph=graph) as sess:
        # Restore variables from disk.
        saver.restore(sess, fn_model)
        print("Model restored.")
        submission = dict()
        k_batch = 0
        for (batch_x, batch_y) in batch_gen:
            if k_batch % 100 == 0:
                logging.info(str(k_batch))
            prediction = sess.run([pred], feed_dict={x: batch_x, keep_prob: 1.0})
            for k,p in enumerate(prediction[0]):
                fname, label = batch_y[k].decode(), id2name[p]
                submission[fname] = label
            k_batch += 1


        with open(os.path.join('assets/corpora/corpus7/', 'submission_test.csv'), 'w') as fout:
            fout.write('fname,label\n')
            for fname, label in submission.items():
                fout.write('{},{}\n'.format(fname, label))
