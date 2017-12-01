from create_soundcorpus import SC_Config, SoundCorpusCreator
import tensorflow as tf
import numpy as np
from glob import glob
from batch_gen import SoundCorpus
from architectures import Model2 as Model
import os
import pickle
import logging
from python_speech_features import mfcc, delta


logging.basicConfig(level=logging.INFO)

class Config:
    soundcorpus_dir = 'assets/corpora/corpus12/'
    batch_size = 287
    is_training = False
    use_batch_norm = False
    keep_prob = 1
    display_step = 10



cfg = Config()
silence_corpus = SoundCorpus(cfg.soundcorpus_dir, mode = 'silence')
test_corpus = SoundCorpus(cfg.soundcorpus_dir,mode='own_test',fn='own_test.p.soundcorpus.p')

#batch_gen = test_corpus.batch_gen(1, do_mfcc=True)
batch = []
try:
    for item in test_corpus:

        batch.append(item)
except:
    pass


decoder = silence_corpus.decoder
encoder = silence_corpus.encoder
num_classes=len(decoder)

model = Model(cfg)
# set_graph Graph

batch_size = cfg.batch_size
is_training = cfg.is_training

def _do_mfcc(signal):
    signal = mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                  nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0,
                  ceplifter=0, appendEnergy=True)
    dsignal = delta(signal, N=1)
    ddsignal = delta(dsignal, N=1)
    signal = np.stack([signal, dsignal, ddsignal], axis=2)
    return signal

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
silence_ids = [id for id,item in enumerate(batch) if sum(abs(item['wav'])) < 16000]
batch_x = [_do_mfcc(item['wav']) for item in batch]
batch_y = [item['label'] for item in batch]
def predict():
    fn_model = 'models/model21/model_mfcc_bsize256_e2.ckpt'
    # %%


    with tf.Session(graph=graph) as sess:
        # Restore variables from disk.
        saver.restore(sess, fn_model)
        print("Model restored.")
        submission = dict()
        k_batch = 0
        predic, cm, acc = sess.run([pred,confusion_matrix, accuracy], feed_dict={x: batch_x, y:batch_y, keep_prob: 1.0})


        print(cm)
        print('Acc',acc)
        k_batch += 1

    return predic, cm, acc


def build_new_test_corpus():
    root = 'assets/new_test/label/'
    sc_cfg = SC_Config('test')
    data = []
    label_folders = [l for l in os.listdir(root) if not l.startswith('.')]
    for folder in label_folders:
        fns = [fn for fn in os.listdir(root + folder + '/') if fn.endswith('.wav')]
        for fn in fns:
            if folder in sc_cfg.possible_labels:
                label_id = sc_cfg.name2id[folder]
            else:
                label_id = sc_cfg.name2id['unknown']
            data.append((label_id,'',root + folder + '/' + fn))

    np.random.shuffle(data)
    portion_unknown = 0.09
    portion_silence = 0.09
    new_data = [data[0]]
    for d in data[1:]:
        if d[0] == sc_cfg.name2id['unknown']:
            if len([d for d in new_data if d[0] == sc_cfg.name2id['unknown']])/len(new_data) < portion_unknown:
                new_data.append(d)
        elif d[0] == sc_cfg.name2id['silence']:
            if len([d for d in new_data if d[0] == sc_cfg.name2id['silence']])/len(new_data) < portion_silence:
                new_data.append(d)
        else:
            new_data.append(d)
    np.random.shuffle(new_data)
    test_corpus = SoundCorpusCreator(sc_cfg)
    corpus = []
    for d in new_data:
        label_id = d[0]
        signal = test_corpus._read_wav_and_pad(d[2]) #rather static function
        corpus.append(dict(label=np.int32(label_id),wav=signal,))
    print(len(corpus))
    save_name = sc_cfg.save_dir
    save_name += 'own_test' + '.'
    save_name += ''.join(['p'])
    save_name += '.soundcorpus.p'
    logging.info('saving under: ' + save_name)
    with open(save_name, 'wb') as f:
        pickler = pickle.Pickler(f)
        for e in corpus:
            pickler.dump(e)
    return len(corpus)


    #  display dict
#if __name__ == '__main__':

prediction, cm, acc = predict()

acc_dict = {}
for c in range(num_classes):
    acc_id = cm[c,c]/sum(cm[c,:])
    acc_dict[decoder[c]] = acc_id


print(sum([cm[i,i] for i in range(num_classes-1)])/sum(sum(cm[:-1,:-1])))
for item in acc_dict:
    print(item,acc_dict[item])



