from create_soundcorpus import SC_Config, SoundCorpusCreator
import tensorflow as tf
import numpy as np
from glob import glob
from batch_gen import SoundCorpus
from architectures import Model2 as Model
import os
import pickle
import logging

logging.basicConfig(level=logging.INFO)

class Config:
    soundcorpus_dir = 'assets/corpora/corpus9/'
    batch_size = 249
    is_training = False
    use_batch_norm = False
    keep_prob = 1
    display_step = 10
    logs_path = 'models/model5/logs/'


cfg = Config()

corpus = SoundCorpus(cfg.soundcorpus_dir,mode='train',fn='new_test_train.pm.soundcorpus.p')

batch_gen = corpus.batch_gen(249)

decoder = corpus.decoder
num_classes=len(decoder)

model = Model(cfg)
# set_graph Graph

batch_size = cfg.batch_size
is_training = cfg.is_training


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
        confusion_matrix = tf.confusion_matrix(tf.reshape(y, [-1]),pred,num_classes)

        tf.summary.scalar('accuracy', accuracy)
    saver = tf.train.Saver()

def predict():
    fn_model = 'models/model7/logs13/model_mfcc_bsize256_e9.ckpt'
    # %%
    id2name = corpus.decoder

    with tf.Session(graph=graph) as sess:
        # Restore variables from disk.
        saver.restore(sess, fn_model)
        print("Model restored.")
        submission = dict()
        k_batch = 0
        try:
            for (batch_x, batch_y) in batch_gen:
                if k_batch % 1000 == 0:
                    logging.info(str(k_batch))
                prediction, cm, acc = sess.run([pred,confusion_matrix, accuracy], feed_dict={x: batch_x, y:batch_y, keep_prob: 1.0})
                #for k,p in enumerate(prediction[0]):
                    #print(p)

                print(cm)
                print('Acc',acc)
                k_batch += 1
        except EOFError:
            pass
    return prediction, cm, acc


def build_new_test_corpus():
    sc_cfg = SC_Config()
    sc_cfg.dir_files = 'new_test/label/*/*wav'
    sc_cfg.unknown_portion = 1
    test_corpus = SoundCorpusCreator(sc_cfg)
    sc_cfg.save_dir += 'new_test_'
    test_corpus.build_corpus('train')

#if __name__ == '__main__':

prediction, cm, acc = predict()

acc_dict = {}
for c in range(num_classes):
    acc_id = cm[c,c]/sum(cm[:,c])
    acc_dict[decoder[c]] = acc_id


print(sum([cm[i,i] for i in range(num_classes-1)])/sum(sum(cm[:-1,:-1])))
