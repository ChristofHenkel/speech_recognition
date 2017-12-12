from silence_svm import *
import glob
import os
from input_features import stacked_mfcc
import time
from architectures import BaselineSilence2 as Baseline
import tensorflow as tf
from silence_detection import SilenceDetector

silence_pathnames = 'assets/data_augmentation/silence/background/*.wav'
all_train_pathnames = 'assets/train/audio/*/*wav'
bn_dir = "assets/train/audio/_background_noise_/"
silence_fnames = glob.glob(silence_pathnames)
speech_fnames = [x for x in glob.glob(all_train_pathnames) if
                 os.path.dirname(x) is not bn_dir]

##################### Load data set
X_train, X_test, y_train, y_test, ss = get_balanced_corpus(silence_fnames,speech_fnames,8000,0.5, is_split=True)
X_train = np.asarray([stacked_mfcc(x) for x in X_train])
X_test = np.asarray([stacked_mfcc(x) for x in X_test])

print('Input dims: ')
print(X_train.shape)

from batch_gen import SoundCorpus
import pickle

with open('assets/corpora/corpus12/' + 'fname2label.p', 'rb') as f:
    fname2label = pickle.load(f)
test_corpus = SoundCorpus('assets/corpora/corpus12/', mode='own_test', fn='own_test_fname.p.soundcorpus.p')
SC = SilenceDetector()

test_data = [d for d in test_corpus if not SC.is_silence(d['wav'])]
X_own_test = [stacked_mfcc(d['wav']) for d in test_data]
y_own_test = [0 if fname2label[d['label']] == 11 else 1 for d in test_data]

decoder = {0: 'silence',
           1: 'speech'}

num_classes = 2
graph = tf.Graph()
baseline = Baseline()


# H Params
#learning_rate = 0.005
#epochs = 40
#keep_probability = 0.9
#momentum = 0.1
#lr_decay_rate = 0.8
#lr_change_steps = 10
# gardiendecent
# BAselineSilence

learning_rate = 0.01
epochs = 41
keep_probability = 0.9
momentum = 0.1
lr_decay_rate = 0.8
lr_change_steps = 10


#display Params
logging.info('Setting Graph Variables')
tf.reset_default_graph()
with graph.as_default():
    # tf Graph input
    tf.set_random_seed(1)
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, shape=(None, 99, 13, 3), name="input")
        y = tf.placeholder(tf.int64, shape=(None,), name="input")
        keep_prob = tf.placeholder(tf.float32, name="dropout")

    with tf.variable_scope('logit'):
        logits = baseline.calc_logits(x, keep_prob, num_classes)

    with tf.variable_scope('costs'):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        cost = tf.reduce_mean(xent, name='xent')
        tf.summary.scalar('cost', cost)

    with tf.variable_scope('acc'):
        pred = tf.argmax(logits, 1)
        correct_prediction = tf.equal(pred, tf.reshape(y, [-1]))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accu')
        confusion_matrix = tf.confusion_matrix(tf.reshape(y, [-1]), pred, num_classes)
        tf.summary.scalar('accuracy', accuracy)
    with tf.variable_scope('acc_per_class'):
        for i in range(num_classes):
            acc_id = confusion_matrix[i, i] / tf.reduce_sum(confusion_matrix[i, :])
            tf.summary.scalar(decoder[i], acc_id)

    # train ops
    gradients = tf.gradients(cost, tf.trainable_variables())
    tf.summary.scalar('grad_norm', tf.global_norm(gradients))
    # gradients, _ = tf.clip_by_global_norm(raw_gradients,max_gradient, name="clip_gradients")
    # gradnorm_clipped = tf.global_norm(gradients)
    # tf.summary.scalar('grad_norm_clipped', gradnorm_clipped)
    iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)
    lr_ = tf.Variable(learning_rate, dtype=tf.float64, name="lr_", trainable=False)
    decay = tf.Variable(lr_decay_rate, dtype=tf.float64, name="decay", trainable=False)
    steps_ = tf.Variable(lr_change_steps, dtype=tf.int64, name="setps_", trainable=False)
    lr = tf.train.exponential_decay(lr_, iteration, steps_, decay, staircase=True)
    #tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).apply_gradients(
        zip(gradients, tf.trainable_variables()),
        name="train_step",
        global_step=iteration)

    saver = tf.train.Saver(max_to_keep=5)
    summaries = tf.summary.merge_all()

logging.info('Done')


def train():
    with tf.Session(graph=graph) as sess:
        logging.info('Start training')
        init = tf.global_variables_initializer()
        sess.run(init)
        global_step = 0

        for epoch in range(epochs):

            step = 1

            # Keep training until reach max iterations
            current_time = time.time()



                # Run optimization op (backprop)
            summary_, _ = sess.run([summaries, optimizer],
                                   feed_dict={x: X_train, y: y_train,
                                              keep_prob: keep_probability})
                #train_writer.add_summary(summary_, global_step)
                #if step % print_step == 0:
                    # Calculate batch accuracy

            logging.info('epoch %s - step %s' % (epoch, step))

            current_time = time.time()
            c, acc, cm = sess.run([cost, accuracy, confusion_matrix],
                                  feed_dict={x: X_train, y: y_train,
                                             keep_prob: keep_probability})

            print(c, acc)
            print(cm)
            c_val, acc_val, cm_val = sess.run([cost, accuracy, confusion_matrix],feed_dict={x: X_test, y: y_test,
                                                                                            keep_prob: 1})
            print(c_val, acc_val)
            print(cm_val)
            step += 1
            global_step += 1
            if epoch % 5 == 0:
                c_val_own, acc_val_own, cm_val_own = sess.run([cost, accuracy, confusion_matrix],
                                                  feed_dict={x: X_own_test, y: y_own_test,
                                                             keep_prob: 1})
                print("validation:", c_val_own, acc_val_own)
                print(cm_val_own)
            # if epoch % cfg.epochs_per_save == 0:
            #self.save(sess, epoch)
            #val_batch_gen = self.valid_corpus.batch_gen(self.len_valid, do_mfcc=True)
            #val_batch_x, val_batch_y = next(val_batch_gen)

            #valid_writer.add_summary(summary_val, global_step)
            #print("validation:", c_val, acc_val)

        print("Optimization Finished!")
        model_name = 'model__e%s.ckpt' %epoch
        s_path = saver.save(sess, 'models/model_silence2/' + model_name)
        print("Model saved in file: %s" % s_path)

        #self.result = [['train_acc', acc], ['val_acc', acc_val]]

train()




#with tf.Session(graph=graph) as sess:
#    saver.restore(sess, 'models/model_silence/model__e39.ckpt')
#    logging.info('Start training')
