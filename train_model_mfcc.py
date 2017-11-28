"""
# Good MFCC explanation:
http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""

from batch_gen import SoundCorpus, BatchGenerator
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import time
import logging
import pickle
import os
from architectures import Model2 as Model
logging.basicConfig(level=logging.DEBUG)


class Config:
    soundcorpus_dir = 'assets/corpora/corpus11/'
    batch_size = 256
    is_training = True
    use_batch_norm = True
    keep_prob = 0.5
    max_gradient = 5
    tf_seed = 4
    learning_rate = 1
    display_step = 10
    epochs = 5
    logs_path = 'models/model0/'

cfg = Config()

corpus = SoundCorpus(cfg.soundcorpus_dir,mode='train')
valid_corpus = SoundCorpus(cfg.soundcorpus_dir, mode = 'valid')

bg_corpus = SoundCorpus(cfg.soundcorpus_dir,mode='train', fn = 'background.p.soundcorpus.p')
unknown_corpus = SoundCorpus(cfg.soundcorpus_dir,mode='train', fn = 'unknown.p.soundcorpus.p')
silence_corpus = SoundCorpus(cfg.soundcorpus_dir,mode='silence', fn = 'silence.p.soundcorpus.p')
advanced_gen = BatchGenerator(corpus,bg_corpus,unknown_corpus,silence_corpus, cfg.batch_size)





decoder = corpus.decoder
num_classes=len(decoder)

# set_graph Graph

batch_size = cfg.batch_size
is_training = cfg.is_training
max_gradient = cfg.max_gradient

training_iters = corpus.len

cnn_model = Model(cfg)
graph = tf.Graph()
with graph.as_default():
    # tf Graph input
    tf.set_random_seed(cfg.tf_seed)
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, shape=(None, 99, 13, 3), name="input")
        y = tf.placeholder(tf.int64, shape=(None,), name="input")
        keep_prob = tf.placeholder(tf.float32, name="dropout")



    # (128, 12) -> (1)
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    with tf.variable_scope('logit'):
        logits = cnn_model.calc_logits(x,keep_prob,num_classes)
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
    with tf.variable_scope('acc_per_class'):
        for i in range(num_classes):
            acc_id = confusion_matrix[i,i]/tf.reduce_sum(confusion_matrix[i,:])
            tf.summary.scalar(corpus.decoder[i], acc_id)


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
        batch_gen = corpus.batch_gen(cfg.batch_size)
        batch_x, batch_y = next(batch_gen)
        cm,l, acc, pred_,y_, id1_ = sess.run([confusion_matrix,logits,accuracy,pred,y,id1], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})
        print(cm)
        print(l, acc)
        return cm, l, acc,pred_,y_, id1_



def train_model():
    with tf.Session(graph=graph) as sess:
        logging.info('Start training')
        init = tf.global_variables_initializer()
        train_writer = tf.summary.FileWriter(cfg.logs_path, graph=graph)
        sess.run(init)
        global_step = 0

        batch_gen = advanced_gen.batch_gen()
        for epoch in range(cfg.epochs):
            step = 1

            # Keep training until reach max iterations
            current_time = time.time()

            while step * batch_size < training_iters:
                #for (batch_x,batch_y) in batch_gen:
                batch_x, batch_y = next(batch_gen)
                # logging.info('epoch ' + str(epoch) + ' - step ' + str(step))
                #batch_x, batch_y = next(gen.batch_gen())

                # Run optimization op (backprop)
                summary_, _ = sess.run([summaries,optimizer], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})
                train_writer.add_summary(summary_, global_step)
                if step % cfg.display_step == 0:
                    # Calculate batch accuracy
                    logging.info('epoch ' + str(epoch) + ' - step ' + str(step))
                    logging.info('runtime for batch of ' + str(cfg.batch_size * cfg.display_step) + ' ' + str(time.time()-current_time))
                    current_time = time.time()
                    c, acc, cm= sess.run([cost, accuracy,confusion_matrix], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})

                    print(c, acc)
                    print(cm)
                step += 1
                global_step += 1
            print('saving model...', end='')
            model_name = 'model_%s_bsize%s_e%s.ckpt' % ('mfcc',batch_size,epoch)

            s_path = saver.save(sess, cfg.logs_path + model_name)
            print("Model saved in file: %s" % s_path)
            #val_batch_gen = valid_corpus.batch_gen(2000)
            #val_batch_x, val_batch_y = next(val_batch_gen)
            #c_val, acc_val = sess.run([cost, accuracy], feed_dict={x: val_batch_x, y: val_batch_y, keep_prob: 1})

            #print(c_val, acc_val)

        print("Optimization Finished!")

def predict():
    fn_model = 'models/model6/logs4/model_mfcc_bsize256_e4.ckpt'
    # %%
    id2name = corpus.decoder

    batch_gen = corpus.batch_gen(6)
    with tf.Session(graph=graph) as sess:
        # Restore variables from disk.
        saver.restore(sess, fn_model)
        print("Model restored.")
        predictions = []
        k_batch = 0
        try:
            for (batch_x, batch_y) in batch_gen:
                if k_batch % 100 == 0:
                    print('------')
                    logging.info(str(k_batch))
                prediction = sess.run([pred], feed_dict={x: batch_x, keep_prob: 1.0})
                print(prediction)
                for k,p in enumerate(prediction[0]):
                    label_true, label = id2name[batch_y[k]], id2name[p]
                    predictions.append([label_true,label])
                k_batch += 1
        except EOFError:
            pass

if __name__ == '__main__':
    #cm, l, acc, pred_,y_, id1_ = debug_model()
    train_model()

