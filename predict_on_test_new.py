from create_soundcorpus import SC_Config, SoundCorpusCreator
import tensorflow as tf
import numpy as np
from glob import glob
from batch_gen import SoundCorpus
from architectures import Model5 as Model
import os
import pickle
import logging
from python_speech_features import mfcc, delta
from input_features import stacked_mfcc
from silence_detection import SilenceDetector


logging.basicConfig(level=logging.INFO)

class Config:
    soundcorpus_dir = 'assets/corpora/corpus12/'
    is_training = True
    use_batch_norm = True
    keep_prob = 1
    display_step = 10

cfg = Config()
silence_corpus = SoundCorpus(cfg.soundcorpus_dir, mode = 'silence')
test_corpus = SoundCorpus(cfg.soundcorpus_dir,mode='own_test',fn='own_test.p.soundcorpus.p')
silence_classifier = SilenceDetector()

batch = [item for item in test_corpus]

decoder = silence_corpus.decoder
encoder = silence_corpus.encoder
num_classes=len(decoder) - 1

model = Model(cfg)
# set_graph Graph

# batch_size = cfg.batch_size
is_training = cfg.is_training



graph = tf.Graph()
with graph.as_default():
    # tf Graph input
    tf.set_random_seed(3)
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, shape=(None, 99, 13, 3), name="input")
        # x.set_shape([batch_size, size])
        #y = tf.placeholder(tf.int64, shape=(None,), name="input")
        keep_prob = tf.placeholder(tf.float32, name="dropout")


    with tf.variable_scope('logit'):
        logits = model.calc_logits(x, keep_prob, num_classes)

    with tf.variable_scope('acc'):
        pred = tf.argmax(logits, 1)
        #correct_prediction = tf.equal(pred, tf.reshape(y, [-1]))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accu')
        #confusion_matrix = tf.confusion_matrix(tf.reshape(y, [-1]),pred,num_classes)

        #tf.summary.scalar('accuracy', accuracy)
    saver = tf.train.Saver()




true_silence_ids = [id for id,item in enumerate(batch) if item['label'] == 11]
silence_ids = [id for id,item in enumerate(batch) if silence_classifier.is_silence(item['wav'])]

batch2 = [b for id, b in enumerate(batch) if id not in true_silence_ids]
batch_x = [stacked_mfcc(item['wav']) for item in batch2]
batch_y = [item['label'] for item in batch2]
batch_x_incl_silence = [stacked_mfcc(item['wav']) for item in batch]
batch_y_incl_silence = [item['label'] for item in batch]

batch_y_test = [np.random.randint(0,11) for b in batch_y]
batch_y_incl_silence_test = [np.random.randint(0,11) for b in batch_y_incl_silence]

def predict(batch_x,batch_y):
    # fn_model = 'models/model40/model_mfcc_bsize512_e49.ckpt'
    fn_model = 'models/model48/model_mfcc_bsize512_e49.ckpt'
    # %%


    with tf.Session(graph=graph) as sess:
        # Restore variables from disk.
        saver.restore(sess, fn_model)
        print("Model restored.")
        k_batch = 0
        predic = sess.run([pred],feed_dict={x: batch_x_incl_silence, keep_prob: 1.0})
        #predic, cm, acc = sess.run([pred,confusion_matrix, accuracy], feed_dict={x: batch_x_incl_silence, y:batch_y_incl_silence_test, keep_prob: 1.0})
        counter = 0
        predic_incl_silence = predic[0]
        for k, p in enumerate(predic[0]):
            if silence_classifier.is_silence(batch[k]['wav']):
                #counter +=1
                predic_incl_silence[k] = 11
                print(predic_incl_silence[k] == batch_y_incl_silence[k])
        print(counter)
        bool_acc = [predic_incl_silence[k] == batch_y_incl_silence[k] for k,_ in enumerate(batch_y_incl_silence)]
        acc = bool_acc.count(True)/len(batch_y_incl_silence)
        #print(cm)
        #print('Acc',acc)
        k_batch += 1

    return predic_incl_silence


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
            data.append((label_id,fn,root + folder + '/' + fn))

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
    fname2label = {}
    corpus = []
    for d in new_data:
        label_id = d[0]
        fname = d[1]
        signal = test_corpus._read_wav_and_pad(d[2]) #rather static function
        corpus.append(dict(label=np.str(fname),wav=signal,))
        fname2label[fname] = label_id
    print(len(corpus))
    with open(sc_cfg.save_dir + 'fname2label.p', 'wb') as f:
        pickle.dump(fname2label,f)

    save_name = sc_cfg.save_dir
    save_name += 'own_test_fname' + '.'
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

#prediction, acc = predict(batch_x,batch_y)
prediction = predict(batch_x,batch_y)
#prediction = prediction[0]
#acc2 = [prediction[k] == batch_y[k] for k,_ in enumerate(batch_y)].count(True)/len(batch_y)
acc3 = [prediction[k] == batch_y_incl_silence[k] for k,_ in enumerate(batch_y_incl_silence)].count(True)/len(batch_y_incl_silence)
#acc_dict = {}
#for c in range(num_classes):
#    acc_id = cm[c,c]/sum(cm[c,:])
#    acc_dict[decoder[c]] = acc_id


#print(sum([cm[i,i] for i in range(num_classes)])/sum(sum(cm)))
#print(' ')
#for item in acc_dict:
#    print(item,acc_dict[item])



