"""
try regulizer:

regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
layer2 = tf.layers.conv2d(
    inputs,
    filters,
    kernel_size,
    kernel_regularizer=regularizer)
"""
from batch_gen import SoundCorpus, BatchGenerator
import pickle
import tensorflow as tf
import time
import logging
import os
import csv

from architectures import Baseline8 as Baseline
logging.basicConfig(level=logging.DEBUG)


class Config:
    soundcorpus_dir = 'assets/corpora/corpus12/'
    model_name = 'model68'
    logs_path = 'models/' + model_name + '/'
    max_ckpt_to_keep = 10


class Hparams:
    is_training = True
    use_batch_norm = False
    keep_prob = 0.9
    max_gradient = 5
    tf_seed = 5
    learning_rate = 0.1
    lr_decay_rate = 0.9
    lr_change_steps = 100
    epochs = 50
    epochs_per_save = 1
    momentum = 0.2


class DisplayParams:
    print_step = 10
    print_step_val = 500
    print_confusion_matrix = True


class BatchParams:
    batch_size = 512
    do_mfcc = True  # batch will have dims (batch_size, 99, 13, 3)
    dims_mfcc = (99, 13, 3)
    portion_unknown = 0.09
    portion_silence = 0
    portion_noised = 1
    lower_bound_noise_mix = 0.5
    upper_bound_noise_mix = 1
    noise_unknown = True
    noise_silence = True


class Model:

    def __init__(self):
        self.cfg = Config()
        self.h_params = Hparams()
        self.batch_params = BatchParams()
        self.display_params = DisplayParams()

        if not os.path.exists(self.cfg.logs_path):
            os.makedirs(self.cfg.logs_path)
        self.write_config()

        self.graph = tf.Graph()
        self.tf_seed = tf.set_random_seed(self.h_params.tf_seed)

        self.baseline = Baseline(self.h_params)
        self.infos = self._load_infos()
        self.train_corpus = SoundCorpus(self.cfg.soundcorpus_dir, mode='train')
        self.valid_corpus = SoundCorpus(self.cfg.soundcorpus_dir, mode='valid', fn='valid.p.soundcorpus.p')
        self.len_valid = self.valid_corpus._get_len()
        self.noise_corpus = SoundCorpus(self.cfg.soundcorpus_dir, mode='background', fn='background.p.soundcorpus.p')
        self.unknown_corpus = SoundCorpus(self.cfg.soundcorpus_dir, mode='unknown', fn='unknown.p.soundcorpus.p')
        #self.silence_corpus = SoundCorpus(self.cfg.soundcorpus_dir, mode='silence', fn='silence.p.soundcorpus.p')


        self.advanced_gen = BatchGenerator(self.batch_params, self.train_corpus, self.noise_corpus, self.unknown_corpus,
                                      SilenceCorpus=None)

        self.encoder = self.infos['name2id']
        self.decoder = self.infos['id2name']
        self.num_classes = len(self.decoder) - 1 #11
        self.training_iters = self.train_corpus.len
        self.result = None

    def _load_infos(self):
        with open(self.cfg.soundcorpus_dir + 'infos.p', 'rb') as f:
            infos = pickle.load(f)
        return infos

    def save(self, sess, epoch):
        print('saving model...', end='')
        model_name = 'model_%s_bsize%s_e%s.ckpt' % ('mfcc', self.batch_params.batch_size, epoch)
        s_path = self.saver.save(sess, self.cfg.logs_path + model_name)
        print("Model saved in file: %s" % s_path)

    @staticmethod
    def class2list(class_):
        class_list = [[item,class_.__dict__ [item]]for item in sorted(class_.__dict__ ) if not item.startswith('__')]
        return class_list

    def get_config(self):
        config_list = []
        for line in self.class2list(Config):
            config_list.append(line)
        for line in self.class2list(Hparams):
            config_list.append(line)
        for line in self.class2list(DisplayParams):
            config_list.append(line)
        for line in self.class2list(BatchParams):
            config_list.append(line)
        return config_list

    def add_experiment_to_csv(self):
        with open('model_runs.csv','a') as csvfile:
            writer = csv.writer(csvfile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            config_list = self.get_config()
            result_list = self.result
            #writer.writerow([c[0] for c in config_list])
            writer.writerow([c[1] for c in config_list] + [r[1] for r in result_list])


    def write_config(self):
        with open(os.path.join(self.cfg.logs_path, 'config.txt'), 'w') as f:
            f.write('Config\n')
            for line in self.class2list(Config):
                f.write('{} = {}\n'.format(line[0], line[1]))
            f.write('\n')
            f.write('HParams\n')
            for line in self.class2list(Hparams):
                f.write('{} = {}\n'.format(line[0], line[1]))
            f.write('\n')
            f.write('DisplayParams\n')
            for line in self.class2list(DisplayParams):
                f.write('{} = {}\n'.format(line[0], line[1]))
            f.write('\n')
            f.write('BatchParams\n')
            for line in self.class2list(BatchParams):
                f.write('{} = {}\n'.format(line[0], line[1]))

    def restore(self, sess, fn_model):
        self.saver.restore(sess, fn_model)
        print("Model restored.")


    def predict(self, batch_x_iter, fn_model):
        with tf.Session(graph=self.graph) as sess:

            self.restore(sess, fn_model)
            predictions = []
            k_batch = 0
            try:
                for batch_x in batch_x_iter:
                    if k_batch % 100 == 0:
                        logging.info(str(k_batch))
                    prediction = sess.run([self.pred], feed_dict={self.x: batch_x, self.keep_prob: 1.0})
                    print(prediction)
                    for k,p in enumerate(prediction[0]):
                        predictions.append([batch_x[k],self.decoder[p]])
                    k_batch += 1
            except EOFError:
                pass
        return predictions

    def train(self):
        with tf.Session(graph=self.graph) as sess:
            logging.info('Start training')
            self.init = tf.global_variables_initializer()
            train_writer = tf.summary.FileWriter(self.cfg.logs_path + 'train/', graph=self.graph)
            valid_writer = tf.summary.FileWriter(self.cfg.logs_path + 'valid/')
            sess.run(self.init)
            global_step = 0

            batch_gen = self.advanced_gen.batch_gen()
            for epoch in range(self.h_params.epochs):

                step = 1

                # Keep training until reach max iterations
                current_time = time.time()

                while step * self.batch_params.batch_size < self.training_iters:
                    # for (batch_x,batch_y) in batch_gen:
                    batch_x, batch_y = next(batch_gen)
                    # logging.info('epoch ' + str(epoch) + ' - step ' + str(step))
                    # batch_x, batch_y = next(gen.batch_gen())

                    # Run optimization op (backprop)
                    summary_, _ = sess.run([self.summaries, self.optimizer],
                                           feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.h_params.keep_prob})
                    train_writer.add_summary(summary_, global_step)
                    if step % self.display_params.print_step == 0:
                        # Calculate batch accuracy

                        logging.info('epoch %s - step %s' % (epoch, step))
                        logging.info('runtime for batch of ' + str(self.batch_params.batch_size * self.display_params.print_step) + ' ' + str(
                            time.time() - current_time))
                        current_time = time.time()
                        c, acc, cm = sess.run([self.cost, self.accuracy, self.confusion_matrix],
                                              feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.h_params.keep_prob})

                        print(c, acc)
                        if self.display_params.print_confusion_matrix:
                            print(cm)
                        print(self.advanced_gen.batches_counter)


                    step += 1
                    global_step += 1
                # if epoch % cfg.epochs_per_save == 0:
                self.save(sess, epoch)
                val_batch_gen = self.valid_corpus.batch_gen(self.len_valid, do_mfcc=True)
                val_batch_x, val_batch_y = next(val_batch_gen)
                summary_val, c_val, acc_val = sess.run([self.summaries, self.cost, self.accuracy],
                                                       feed_dict={self.x: val_batch_x, self.y: val_batch_y,
                                                                  self.keep_prob: 1})
                valid_writer.add_summary(summary_val, global_step)
                print("validation:", c_val, acc_val)

            print("Optimization Finished!")
            self.result = [['train_acc',acc],['val_acc',acc_val]]
        pass

    #def debug(self):
    #    with tf.Session(graph=graph) as sess:
    #        init = tf.global_variables_initializer()
    #        sess.run(init)
    #        batch_gen = advanced_gen.batch_gen()
    #        batch_x, batch_y = next(batch_gen)
    #        l, kw = sess.run([logits, krw], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})
    #        return l, kw
    #    pass

    def set_graph(self):
        logging.info('Setting Graph Variables')
        with self.graph.as_default():
            # tf Graph input

            with tf.name_scope("Input"):
                self.x = tf.placeholder(tf.float32, shape=(None, 99, 13, 3), name="input")
                self.y = tf.placeholder(tf.int64, shape=(None,), name="input")
                self.keep_prob = tf.placeholder(tf.float32, name="dropout")

            with tf.variable_scope('logit'):
                self.logits = self.baseline.calc_logits(self.x, self.keep_prob, self.num_classes)


            with tf.variable_scope('costs'):
                self.xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
                self.cost = tf.reduce_mean(self.xent, name='xent')

                tf.summary.scalar('cost', self.cost)

            with tf.variable_scope('acc'):
                self.pred = tf.argmax(self.logits, 1)
                self.correct_prediction = tf.equal(self.pred, tf.reshape(self.y, [-1]))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accu')
                self.confusion_matrix = tf.confusion_matrix(tf.reshape(self.y, [-1]), self.pred, self.num_classes)
                tf.summary.scalar('accuracy', self.accuracy)
            with tf.variable_scope('acc_per_class'):
                for i in range(self.num_classes):
                    acc_id = self.confusion_matrix[i, i] / tf.reduce_sum(self.confusion_matrix[i, :])
                    tf.summary.scalar(self.decoder[i], acc_id)

            # train ops
            self.gradients = tf.gradients(self.cost, tf.trainable_variables())
            tf.summary.scalar('grad_norm', tf.global_norm(self.gradients))
            # gradients, _ = tf.clip_by_global_norm(raw_gradients,max_gradient, name="clip_gradients")
            # gradnorm_clipped = tf.global_norm(gradients)
            # tf.summary.scalar('grad_norm_clipped', gradnorm_clipped)
            self.iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)
            self.lr_ = tf.Variable(self.h_params.learning_rate, dtype=tf.float64, name="lr_", trainable=False)
            decay = tf.Variable(self.h_params.lr_decay_rate, dtype=tf.float64, name="decay", trainable=False)
            steps_ = tf.Variable(self.h_params.lr_change_steps, dtype=tf.int64, name="setps_", trainable=False)
            self.lr = tf.train.exponential_decay(self.lr_, self.iteration,steps_, decay, staircase=True)
            tf.summary.scalar('learning_rate', self.lr)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.h_params.momentum).apply_gradients(
                zip(self.gradients, tf.trainable_variables()),
                name="train_step",
                global_step=self.iteration)

            self.saver = tf.train.Saver(max_to_keep=self.cfg.max_ckpt_to_keep)
            self.summaries = tf.summary.merge_all()

        logging.info('Done')


if __name__ == '__main__':

    m = Model()
    m.set_graph()
    m.train()