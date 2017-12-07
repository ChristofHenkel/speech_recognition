import tensorflow as tf
from batch_gen import SoundCorpus
from architectures import Model5 as Model
import os
import logging
from silence_detection import SilenceDetector
from input_features import stacked_mfcc
import pickle
logging.basicConfig(level=logging.INFO)

class Config:
    soundcorpus_dir = 'assets/corpora/corpus12/'
    batch_size = 298
    is_training = True
    use_batch_norm = True
    keep_prob = 1

cfg = Config()

test_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='test', fn='test.p.soundcorpus.p')
silence_corpus = SoundCorpus(cfg.soundcorpus_dir, mode = 'silence', fn='silence.p.soundcorpus.p')
SC = SilenceDetector()
corpus_len = test_corpus._get_len()


batch_gen = test_corpus.batch_gen(cfg.batch_size)
#batch = [b for b in test_corpus]
decoder = silence_corpus.decoder
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

def prepare_submission(fn_model,fn_out=None):
    #batch_x = [b['wav'] for b in batch]
    #batch_y = [b['label'] for b in batch]

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
                        #fname, label = batch_y[k].decode(), 'silence'
                        fname, label = batch_y[k], 'silence'
                    else:
                        #fname, label = batch_y[k].decode(), decoder[p]
                        fname, label = batch_y[k], decoder[p]
                    submission[fname] = label
                k_batch += 1
        except EOFError:
            pass

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
if __name__ == '__main__':
    submission = prepare_submission(fn_model='models/model40/model_mfcc_bsize512_e49.ckpt',fn_out='submission_test7.csv')
