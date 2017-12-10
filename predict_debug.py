import tensorflow as tf
from batch_gen import SoundCorpus
from architectures import Baseline7 as Model
import os
import logging
from silence_detection import SilenceDetector
from input_features import stacked_mfcc
import pickle
logging.basicConfig(level=logging.INFO)

class Config:
    soundcorpus_dir = 'assets/corpora/corpus12/'
    batch_size = 1
    is_training = False
    use_batch_norm = False
    keep_prob = 1


cfg = Config()
own_test_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='own_test', fn='own_test_fname.p.soundcorpus.p')
test_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='test', fn='test.p.soundcorpus.p')
silence_corpus = SoundCorpus(cfg.soundcorpus_dir, mode = 'silence', fn='silence.p.soundcorpus.p')
SC = SilenceDetector()
own_test_corpus_len = own_test_corpus._get_len()
test_corpus_len = test_corpus._get_len()
if cfg.batch_size > 1:
    rest = test_corpus_len % cfg.batch_size
else:
    rest = 0
num_batches_test = (test_corpus_len - rest) / cfg.batch_size
print('calculated number of batches test: %s' %num_batches_test)
num_batches_test = int(num_batches_test)

num_batches_own_test = (own_test_corpus_len - rest) / cfg.batch_size
print('calculated number of batches test: %s' %num_batches_own_test)
num_batches_own_test = int(num_batches_own_test)

batch_gen_own_test = own_test_corpus.batch_gen(cfg.batch_size)
batch_own_test = [item for item in own_test_corpus]
batch_own_test_x = [item['wav'] for item in batch_own_test]
batch_own_test_y = [item['label'] for item in batch_own_test]

batch_gen_test = test_corpus.batch_gen(cfg.batch_size)
batch_test = [item for item in test_corpus]
batch_test_x = [item['wav'] for item in batch_test]
batch_test_y = [item['label'] for item in batch_test]

if rest > 0:
    rest_batch = [b for b in own_test_corpus][-rest:]
else:
    rest_batch = []
decoder = silence_corpus.decoder
encoder = silence_corpus.encoder
num_classes = len(decoder) -1

model = Model(cfg)

# set_graph Graph
graph = tf.Graph()
#tf.reset_default_graph()
with graph.as_default():
    # tf Graph input
    tf.set_random_seed(3)
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, shape=(None, 99, 13, 3), name="input")
        y = tf.placeholder(tf.int64, shape=(None,), name="input")
        keep_prob = tf.placeholder(tf.float32, name="dropout")
    with tf.variable_scope('logit'):
        logits = model.calc_logits(x, keep_prob, num_classes)
        pred = tf.argmax(logits, 1)
    saver = tf.train.Saver()


fn_model='models/model56/model_mfcc_bsize512_e47.ckpt'



submission_own_test_iter = dict()

with tf.Session(graph=graph) as sess:
    # Restore variables from disk.
    saver.restore(sess, fn_model)
    print("Model restored.")

    for k_batch in range(num_batches_own_test):
        try:
            batch_x, batch_y = [batch_own_test_x[k_batch]], [batch_own_test_y[k_batch]]


            batch_x2 = [stacked_mfcc(b) for b in batch_x]

            if k_batch % 10 == 0:
                logging.info('Batch %s / %s' %(k_batch+1,num_batches_own_test))
            prediction = sess.run([pred], feed_dict={x: batch_x2, keep_prob: 1.0})
            for k,p in enumerate(prediction[0]):
                if SC.is_silence(batch_x[k]):
                    if own_test_corpus.mode == 'test':
                        fname, label = batch_y[k].decode(), 'silence'
                    else:
                        fname, label = batch_y[k], 'silence'
                else:
                    if own_test_corpus.mode == 'test':
                        fname, label = batch_y[k].decode(), decoder[p]
                    else:
                        fname, label = batch_y[k], decoder[p]
                submission_own_test_iter[fname] = label
            #k_batch += 1
        except EOFError:
            pass



submission_own_test_gen = dict()

with tf.Session(graph=graph) as sess:
    # Restore variables from disk.
    saver.restore(sess, fn_model)
    print("Model restored.")

    for k_batch in range(num_batches_own_test):
        try:
            batch_x, batch_y = next(batch_gen_own_test)


            batch_x2 = [stacked_mfcc(b) for b in batch_x]

            if k_batch % 10 == 0:
                logging.info('Batch %s / %s' %(k_batch+1,num_batches_own_test))
            prediction = sess.run([pred], feed_dict={x: batch_x2, keep_prob: 1.0})
            for k,p in enumerate(prediction[0]):
                if SC.is_silence(batch_x[k]):
                    if own_test_corpus.mode == 'test':
                        fname, label = batch_y[k].decode(), 'silence'
                    else:
                        fname, label = batch_y[k], 'silence'
                else:
                    if own_test_corpus.mode == 'test':
                        fname, label = batch_y[k].decode(), decoder[p]
                    else:
                        fname, label = batch_y[k], decoder[p]
                submission_own_test_gen[fname] = label
            #k_batch += 1
        except EOFError:
            pass

a = [p for p in submission_own_test_iter if submission_own_test_iter[p]!=submission_own_test_gen[p]]
print(a)

submission_test_iter = dict()

with tf.Session(graph=graph) as sess:
    # Restore variables from disk.
    saver.restore(sess, fn_model)
    print("Model restored.")

    for k_batch in range(num_batches_test):
        try:
            batch_x, batch_y = [batch_test_x[k_batch]], [batch_test_y[k_batch]]


            batch_x2 = [stacked_mfcc(b) for b in batch_x]

            if k_batch % 10 == 0:
                logging.info('Batch %s / %s' %(k_batch+1,num_batches_test))
            prediction = sess.run([pred], feed_dict={x: batch_x2, keep_prob: 1.0})
            for k,p in enumerate(prediction[0]):
                if SC.is_silence(batch_x[k]):
                    if own_test_corpus.mode == 'test':
                        fname, label = batch_y[k].decode(), 'silence'
                    else:
                        fname, label = batch_y[k], 'silence'
                else:
                    if own_test_corpus.mode == 'test':
                        fname, label = batch_y[k].decode(), decoder[p]
                    else:
                        fname, label = batch_y[k], decoder[p]
                submission_test_iter[fname] = label
            #k_batch += 1
        except EOFError:
            pass



submission_test_gen = dict()

with tf.Session(graph=graph) as sess:
    # Restore variables from disk.
    saver.restore(sess, fn_model)
    print("Model restored.")

    for k_batch in range(num_batches_test):
        try:
            batch_x, batch_y = next(batch_gen_test)


            batch_x2 = [stacked_mfcc(b) for b in batch_x]

            if k_batch % 10 == 0:
                logging.info('Batch %s / %s' %(k_batch+1,num_batches_test))
            prediction = sess.run([pred], feed_dict={x: batch_x2, keep_prob: 1.0})
            for k,p in enumerate(prediction[0]):
                if SC.is_silence(batch_x[k]):
                    if test_corpus.mode == 'test':
                        fname, label = batch_y[k].decode(), 'silence'
                    else:
                        fname, label = batch_y[k], 'silence'
                else:
                    if test_corpus.mode == 'test':
                        fname, label = batch_y[k].decode(), decoder[p]
                    else:
                        fname, label = batch_y[k], decoder[p]
                submission_test_gen[fname] = label
            #k_batch += 1
        except EOFError:
            pass



def get_kth_element(gen,k):
    i = 0
    while i < k:
        _ = next(gen)
        i += 1
    item = next(gen)
    return item



b = [p for p in submission_test_iter if submission_test_iter[p]!=submission_test_gen[p]]
batch_test_y.index(b[0])
print([submission_test_iter[p] for p in b])
print(b)

s = get_kth_element()


def acc():
    with open(cfg.soundcorpus_dir + 'fname2label.p', 'rb') as f:
        fname2label = pickle.load(f)
    comparison = [(p,submission[p],decoder[fname2label[p]]) for p in submission]
    acc = [a[1] == a[2] for a in comparison].count(True)/len(comparison)
    no_silence = [c for c in comparison if c[2] != 'silence']
    acc_dict = {}
    for l in encoder:
        label_part = [c for c in comparison if c[2] == l]
        acc_label = [a[1] == a[2] for a in label_part].count(True)/len(label_part)
        acc_dict[l] = acc_label
    print(acc_dict)

    acc_no_silence = [a[1] == a[2] for a in no_silence].count(True)/len(no_silence)
    print('acc: %s' %acc)
    print('acc w/o silence: %s' % acc_no_silence)
