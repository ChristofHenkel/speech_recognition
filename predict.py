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

class BatchParams:
    batch_size = 512

cfg = Config()
batch_params = BatchParams()
#test_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='own_test', fn='own_test_fname.p.soundcorpus.p')
test_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='test', fn='test.p.soundcorpus.p')
silence_corpus = SoundCorpus(cfg.soundcorpus_dir, mode = 'silence', fn='silence.p.soundcorpus.p')
SC = SilenceDetector()
corpus_len = test_corpus._get_len()
if cfg.batch_size > 1:
    rest = corpus_len % cfg.batch_size
else:
    rest = 0
num_batches = (corpus_len - rest) / cfg.batch_size
print('calculated number of batches: %s' %num_batches)
num_batches = int(num_batches)
batch_gen = test_corpus.batch_gen(cfg.batch_size)
if rest > 0:
    rest_batch = [b for b in test_corpus][-rest:]
else:
    rest_batch = []
decoder = silence_corpus.decoder
encoder = silence_corpus.encoder
num_classes = len(decoder) -1

model = Model(cfg, batch_params)

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

def prepare_submission(fn_model,fn_out=None):
    #batch_x = [b['wav'] for b in batch]
    #batch_y = [b['label'] for b in batch]

    submission = dict()

    with tf.Session(graph=graph) as sess:
        # Restore variables from disk.
        saver.restore(sess, fn_model)
        print("Model restored.")

        #k_batch = 0
        for k_batch in range(num_batches):
            try:
                batch_x, batch_y = next(batch_gen)


                batch_x2 = [stacked_mfcc(b) for b in batch_x]

                if k_batch % 10 == 0:
                    logging.info('Batch %s / %s' %(k_batch+1,num_batches))
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
                    submission[fname] = label
                #k_batch += 1
            except EOFError:
                pass
        for b in rest_batch:
            fname, label = b['label'].decode(), 'unknown'
            submission[fname] = label


        if fn_out is not None:
            with open(os.path.join(cfg.soundcorpus_dir, fn_out), 'w') as fout:
                fout.write('fname,label\n')
                for fname, label in submission.items():
                    fout.write('{},{}\n'.format(fname, label))
    return submission

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
if __name__ == '__main__':
    submission = prepare_submission(fn_model='models/model56/model_mfcc_bsize512_e47.ckpt', fn_out='submission_model56_ckpt_47.csv')
    acc()
    #fn_model = 'models/model47/model_mfcc_bsize512_e49.ckpt'


a = [(p,submission_own_test[p],submission[p]) for p in submission_own_test]