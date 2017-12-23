import tensorflow as tf
from batch_gen import SoundCorpus
from architectures import cnn_one_fpool4_rnn as Baseline
import os
import logging
from silence_detection import SilenceDetector
from input_features import stacked_mfcc, stacked_filterbank
import pickle
import numpy as np
import time
logging.basicConfig(level=logging.INFO)

class Config:
    soundcorpus_dir = 'assets/corpora/corpus14/'
    batch_size = 1
    is_training = False
    use_batch_norm = False
    keep_prob = 1
    test_mode = 'own_test'
    input_transformation = 'filterbank'
    dims_mfcc = (99,26,1)
    do_detect_silence = False
    num_classes = 12
    preprocessed = True


def load_corpus(cfg):
    if cfg.test_mode == 'own_test':
        if cfg.preprocessed is True:
            test_corpus = SoundCorpus('', mode='own_test', fn='own_test_preprocessed.p')
        else:
            test_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='own_test', fn='own_test_fname.p.soundcorpus.p')
    elif cfg.test_mode == 'test':
        test_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='test', fn='test.pf.soundcorpus.p')
    elif cfg.test_mode == 'valid':
        test_corpus = SoundCorpus(cfg.soundcorpus_dir, mode = 'valid', fn = 'valid.p.soundcorpus.p')
    else:
        test_corpus = None
    return test_corpus

def preprocess(test_corpus,cfg, fn_out):
    batch_gen = test_corpus.batch_gen(cfg.batch_size, input_transformation=None)
    num_batches, rest_batch = get_num_batches_rest_batch(test_corpus, cfg.batch_size)
    tic = time.time()
    with open(fn_out, 'wb') as f:
        pickler = pickle.Pickler(f)
        for k_batch in range(num_batches):

            batch_x, batch_y = next(batch_gen)
            batch_x2 = transform_input(batch_x, cfg)

            for k,b in enumerate(batch_x2):
                pickler.dump({'wav':b, 'label':batch_y[k]})
            if k_batch % 50 == 0:
                toc = time.time()
                time_per_date = (toc - tic) / (50 * cfg.batch_size)
                logging.info('Batch %s / %s (%ss/date)' % (k_batch + 1, num_batches, time_per_date))

                tic = time.time()



def get_num_batches_rest_batch(test_corpus, batch_size):
    if cfg.test_mode == 'test':
        corpus_len = 158538
    else:
        corpus_len = test_corpus._get_len()
        test_corpus.reset_gen()
    if batch_size > 1:
        rest = corpus_len % batch_size
    else:
        rest = 0
    num_batches = (corpus_len - rest) / batch_size
    print('calculated number of batches: %s' %num_batches)
    num_batches = int(num_batches)
    if rest > 0:
        test_corpus.reset_gen()
        rest_batch = test_corpus._get_last_n(corpus_len,rest)
    else:
        rest_batch = []
    return num_batches, rest_batch

def load_encoder_decoder(cfg):

    with open(cfg.soundcorpus_dir + 'infos.p','rb') as f:
        content = pickle.load(f)
        decoder = content['id2name']
        encoder = content['name2id']
    return decoder, encoder

cfg = Config()
test_corpus = load_corpus(cfg)
num_batches, rest_batch = get_num_batches_rest_batch(test_corpus,cfg.batch_size)
decoder, encoder = load_encoder_decoder(cfg)

SC = SilenceDetector()
batch_gen = test_corpus.batch_gen(cfg.batch_size, input_transformation=None)


baseline = Baseline(cfg)

# set_graph Graph
graph = tf.Graph()
#tf.reset_default_graph()
with graph.as_default():
    # tf Graph input
    tf.set_random_seed(3)
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, shape=(None,) + cfg.dims_mfcc, name="input")
        y = tf.placeholder(tf.int64, shape=(None,), name="input")
        keep_prob = tf.placeholder(tf.float32, name="dropout")
    with tf.variable_scope('logit'):
        logits = baseline.calc_logits(x, keep_prob, cfg.num_classes)
        pred = tf.argmax(logits, 1)
    saver = tf.train.Saver()

def transform_input(batch_x,cfg):
    if cfg.input_transformation == 'mfcc':
        batch_x2 = np.asarray([stacked_mfcc(b, numcep=cfg.dims_mfcc[1], num_layers=cfg.dims_mfcc[2]) for b in batch_x])
    elif cfg.input_transformation == 'filterbank':
        batch_x2 = np.asarray(
            [stacked_filterbank(b, nfilt=cfg.dims_mfcc[1], num_layers=cfg.dims_mfcc[2]) for b in batch_x])
    else:
        batch_x2 = batch_x
    return batch_x2

def prepare_submission(fn_model,fn_out=None):
    #batch_x = [b['wav'] for b in batch]
    #batch_y = [b['label'] for b in batch]
    if cfg.test_mode in ['test','own_test']:
        submission = dict()
    else:
        submission = []
    with tf.Session(graph=graph) as sess:
        # Restore variables from disk.
        saver.restore(sess, fn_model)
        print("Model restored.")

        tic = time.time()
        for k_batch in range(num_batches):

            try:
                batch_x, batch_y = next(batch_gen)
                if cfg.preprocessed is True:
                    batch_x2 = batch_x

                else:
                    batch_x2 = transform_input(batch_x,cfg)

                if k_batch % 50 == 0:
                    print(batch_x2.shape)
                    print(batch_y.shape)
                    toc = time.time()
                    time_per_date = (toc - tic) / (50 * cfg.batch_size)
                    logging.info('Batch %s / %s (%ss/date)' %(k_batch+1,num_batches,time_per_date))

                    tic = time.time()
                    #time per date
                prediction = sess.run(pred, feed_dict={x: batch_x2, keep_prob: 1.0})
                for k,p in enumerate(prediction):
                    if cfg.do_detect_silence:
                        is_silence = False
                        if not decoder[p] == 'silence':
                            try:
                                is_silence = SC.is_silence2(batch_x[k])
                            except:

                                logging.warning('vad error in file %s - %s' %(k_batch,k))
                        if is_silence:
                            fname, label = batch_y[k], 'silence'
                        else:
                            fname, label = batch_y[k], decoder[p]
                    else:
                        fname, label = batch_y[k], decoder[p]
                    if cfg.test_mode in ['test','own_test']:
                        submission[fname] = label
                    else:
                        submission.append((fname,label))
            except EOFError:
                print('EOFError')
                pass

        if len(rest_batch) > 0:
            rest_batch_x = [b['wav'] for b in rest_batch]
            rest_batch_y = [b['label'] for b in rest_batch]
            rest_batch_x2 = transform_input(rest_batch_x, cfg)
            prediction_rest = sess.run(pred, feed_dict={x: rest_batch_x2, keep_prob: 1.0})
            for k, p in enumerate(prediction_rest):
                fname, label = rest_batch_y[k], decoder[p]
                if cfg.test_mode in ['test', 'own_test']:
                    submission[fname] = label
                else:
                    submission.append((fname, label))


        if fn_out is not None:
            with open(os.path.join(cfg.soundcorpus_dir, fn_out), 'w') as fout:
                fout.write('fname,label\n')
                for fname, label in submission.items():
                    fout.write('{},{}\n'.format(fname, label))
    return submission

def acc(submission):
    with open(cfg.soundcorpus_dir + 'fname2label.p', 'rb') as f:
        fname2label = pickle.load(f)
    if cfg.test_mode in ['test','own_test']:
        comparison = [(p,submission[p],decoder[fname2label[p]]) for p in submission]
    else:
        comparison = [('_', decoder[p[0]], p[1]) for p in submission]
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
if __name__ == '__main__':
    #preprocess(test_corpus, cfg, 'own_test_preprocessed.p')
    submission = prepare_submission(fn_model='models/tmp_model14/model_mfcc_bsize512_e49.ckpt')
    #acc(submission)


