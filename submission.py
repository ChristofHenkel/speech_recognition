import tensorflow as tf
import numpy as np
from glob import glob
from batch_gen import SoundCorpus
from architectures import Model2 as Model
import os
import pickle
import logging
from silence_detection import SilenceDetector
from input_features import stacked_mfcc
logging.basicConfig(level=logging.INFO)

class Config:
    soundcorpus_dir = 'assets/corpora/corpus12/'
    batch_size = 6
    is_training = False
    use_batch_norm = False
    keep_prob = 1
    display_step = 10
    logs_path = 'models/model0/'


cfg = Config()

corpus = SoundCorpus(cfg.soundcorpus_dir, mode='test', fn='test.p.soundcorpus.p')
SC = SilenceDetector()
batch_gen = corpus.batch_gen(cfg.batch_size)

decoder = corpus.decoder
num_classes = len(decoder) -1

model = Model(cfg)
# set_graph Graph

batch_size = cfg.batch_size
is_training = cfg.is_training


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
        logits = model.calc_logits(x, keep_prob, num_classes)
        #predictions = tf.nn.softmax(logits)

    #with tf.variable_scope('costs'):
        #xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #   labels=y, logits=logits)
        #cost = tf.reduce_mean(xent, name='xent')
        # cost += self._decay()

        #tf.summary.scalar('cost', cost)

    #with tf.variable_scope('acc'):
        pred = tf.argmax(logits, 1)
        #correct_prediction = tf.equal(pred, tf.reshape(y, [-1]))
        #accuracy = tf.reduce_mean(
        #    tf.cast(correct_prediction, tf.float32), name='accu')

        #tf.summary.scalar('accuracy', accuracy)
    saver = tf.train.Saver()

def submission():
    fn_model = 'models/model30/model_mfcc_bsize256_e9.ckpt'
    # %%
    id2name = corpus.decoder
    #cfg = Config()
    # cfg.soundcorpus_fp = 'assets/corpora/corpus7/test.pm.soundcorpus.p'
    size = 158538

    submission = dict()

    with tf.Session(graph=graph) as sess:
        # Restore variables from disk.
        saver.restore(sess, fn_model)
        print("Model restored.")

        k_batch = 0
        try:
            for (batch_x, batch_y) in batch_gen:

                batch_x2 = [stacked_mfcc(b) for b in batch_x]

                if k_batch % 1000 == 0:
                    logging.info(str(k_batch))
                prediction = sess.run([pred], feed_dict={x: batch_x2, keep_prob: 1.0})
                for k,p in enumerate(prediction[0]):
                    if SC.is_silence(batch_x[k]):
                        fname, label = batch_y[k].decode(), 'silence'
                    else:
                        fname, label = batch_y[k].decode(), id2name[p]
                    submission[fname] = label
                k_batch += 1
        except EOFError:
            pass

        with open(os.path.join('assets/corpora/corpus12/', 'submission_test.csv'), 'w') as fout:
            fout.write('fname,label\n')
            for fname, label in submission.items():
                fout.write('{},{}\n'.format(fname, label))

if __name__ == '__main__':


    submission()