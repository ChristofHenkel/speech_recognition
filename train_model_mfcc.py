"""
# Good MFCC explanation:
http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""

from batch_gen import SoundCorpus
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import time
import logging
import pickle
import os
from architectures import Model2 as Model1
logging.basicConfig(level=logging.DEBUG)


class Config:
    soundcorpus_dir = 'assets/corpora/corpus7/'
    batch_size = 256
    is_training = True
    use_batch_norm = True
    keep_prob = 0.7
    max_gradient = 5
    learning_rate = 1
    display_step = 10
    epochs = 5
    logs_path = 'models/model6/logs3/'

cfg = Config()

corpus = SoundCorpus(cfg.soundcorpus_dir,mode='train')
valid_corpus = SoundCorpus(cfg.soundcorpus_dir, mode = 'valid')


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



    # (128, 12) -> (1)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    with tf.variable_scope('logit'):
        logits = Model1.calc_logits(x,keep_prob,is_training,cfg.use_batch_norm,num_classes)
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

    # train ops
    gradients, _ = tf.clip_by_global_norm(tf.gradients(cost, tf.trainable_variables()),
                                          max_gradient, name="clip_gradients")
    iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=cfg.learning_rate).apply_gradients(
        zip(gradients, tf.trainable_variables()),
        name="train_step",
        global_step=iteration)


    #pred = tf.argmax(logits, axis=-1)
    #correct_pred = tf.equal(pred, tf.reshape(y, [-1]))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    summaries = tf.summary.merge_all()

# Launch the graph
# TESTING



def debug_model():
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_x, batch_y = next(batch_gen)
        l, acc = sess.run([logits,accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})
        print(l, acc)
        return l, acc



def train_model():
    with tf.Session(graph=graph) as sess:
        logging.info('Start training')
        init = tf.global_variables_initializer()
        train_writer = tf.summary.FileWriter(cfg.logs_path, graph=graph)
        sess.run(init)
        global_step = 0
        val_batch_gen = valid_corpus.batch_gen(1000)
        for epoch in range(cfg.epochs):
            step = 1

            # Keep training until reach max iterations
            current_time = time.time()
            batch_gen = corpus.batch_gen(cfg.batch_size)
            while step * batch_size < training_iters:
                #for (batch_x,batch_y) in batch_gen:
                batch_x, batch_y = next(batch_gen)
                logging.info('epoch ' + str(epoch) + ' - step ' + str(step))
                #batch_x, batch_y = next(gen.batch_gen())

                # Run optimization op (backprop)
                summary_, _ = sess.run([summaries,optimizer], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})
                train_writer.add_summary(summary_, global_step)
                if step % cfg.display_step == 0:
                    # Calculate batch accuracy

                    logging.info('runtime for batch of ' + str(cfg.batch_size * cfg.display_step) + ' ' + str(time.time()-current_time))
                    current_time = time.time()
                    c, acc= sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})

                    print(c, acc)
                step += 1
                global_step += 1
            print('saving model...', end='')
            model_name = 'model_%s_bsize%s_e%s.ckpt' % ('mfcc',batch_size,epoch)

            s_path = saver.save(sess, cfg.logs_path + model_name)
            print("Model saved in file: %s" % s_path)

            val_batch_x, val_batch_y = next(val_batch_gen)
            c, acc = sess.run([cost, accuracy], feed_dict={x: val_batch_x, y: val_batch_y, keep_prob: 1})

            print(c, acc)

        print("Optimization Finished!")



if __name__ == '__main__':
    #l, acc = debug_model()
    train_model()

