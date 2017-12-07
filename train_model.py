
from batch_gen import SoundCorpus, BatchGenerator
import tensorflow as tf
import time
import logging
from architectures import Model5 as Model
logging.basicConfig(level=logging.DEBUG)

class Config:
    soundcorpus_dir = 'assets/corpora/corpus12/'
    is_training = True
    use_batch_norm = True
    keep_prob = 0.8
    max_gradient = 5
    tf_seed = 4
    learning_rate = 1
    lr_decay_rate = 0.9
    lr_change_steps = 100
    display_step = 10
    display_step_val = 50
    epochs = 50
    epochs_per_save = 1
    logs_path = 'models/model42/'

    def save(self):
        with open(self.logs_path + 'config.txt','w') as f:


class BatchParams:
    batch_size = 512
    do_mfcc = True # batch will have dims (batch_size, 99, 13, 3)
    portion_unknown = 0.15
    portion_silence = 0
    portion_noised = 1
    lower_bound_noise_mix = 0.5
    upper_bound_noise_mix = 1
    noise_unknown = False
    noise_silence = True

cfg = Config()

corpus = SoundCorpus(cfg.soundcorpus_dir, mode='train')
valid_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='valid', fn='valid.p.soundcorpus.p')
len_valid = valid_corpus._get_len()
background_noise_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='background', fn='background.p.soundcorpus.p')
unknown_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='unknown', fn='unknown.p.soundcorpus.p')
silence_corpus = SoundCorpus(cfg.soundcorpus_dir, mode='silence', fn='silence.p.soundcorpus.p')

batch_parameters = BatchParams()
advanced_gen = BatchGenerator(batch_parameters, corpus, background_noise_corpus, unknown_corpus, SilenceCorpus=None)

encoder = corpus.encoder
decoder = corpus.decoder

num_classes=len(decoder)-1

# set_graph Graph

batch_size = batch_parameters.batch_size
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

    with tf.variable_scope('logit'):
        logits = cnn_model.calc_logits(x,keep_prob,num_classes)
        predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        cost = tf.reduce_mean(xent, name='xent')#


        tf.summary.scalar('cost', cost)

    with tf.variable_scope('acc'):
        pred = tf.argmax(logits, 1)
        correct_prediction = tf.equal(pred, tf.reshape(y, [-1]))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accu')
        confusion_matrix = tf.confusion_matrix(tf.reshape(y, [-1]),pred,num_classes)
        tf.summary.scalar('accuracy', accuracy)
    with tf.variable_scope('acc_per_class'):
        for i in range(num_classes):
            acc_id = confusion_matrix[i,i]/tf.reduce_sum(confusion_matrix[i,:])
            tf.summary.scalar(corpus.decoder[i], acc_id)



    # train ops
    raw_gradients = tf.gradients(cost, tf.trainable_variables())
    gradnorm = tf.global_norm(raw_gradients)
    tf.summary.scalar('grad_norm', gradnorm)
    gradients, _ = tf.clip_by_global_norm(raw_gradients,max_gradient, name="clip_gradients")
    gradnorm_clipped = tf.global_norm(gradients)
    tf.summary.scalar('grad_norm_clipped', gradnorm_clipped)
    iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)
    lr_ = tf.Variable(cfg.learning_rate, dtype=tf.float64, name="lr_", trainable=False)
    decay = tf.Variable(cfg.lr_decay_rate, dtype=tf.float64, name="decay", trainable=False)
    steps_ = tf.Variable(100, dtype=tf.int64, name="setps_", trainable=False)
    lr = tf.train.exponential_decay(lr_, iteration,steps_, decay, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).apply_gradients(
        zip(gradients, tf.trainable_variables()),
        name="train_step",
        global_step=iteration)


    saver = tf.train.Saver()
    summaries = tf.summary.merge_all()

# Launch the graph
# TESTING



def debug_model():
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_gen = advanced_gen.batch_gen()
        batch_x, batch_y = next(batch_gen)
        l, kw = sess.run([logits, krw], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})
        return l, kw

def aggregate_y(batch_y):
    knowns = [id for id in batch_y if id in [0,1,2,3,4,5,6,7,8,9]]
    unknowns = [id for id in batch_y if id == 11]
    silences = [id for id in batch_y if id == 10]
    len_all = len(knowns) + len(unknowns) + len(silences)
    port_knowns = len(knowns)/len_all
    port_unknowns = len(unknowns)/len_all
    port_silence = len(silences)/len_all
    return dict(known=port_knowns,unknowns=port_unknowns,silence = port_silence, )

def train_model():
    with tf.Session(graph=graph) as sess:
        logging.info('Start training')
        init = tf.global_variables_initializer()
        train_writer = tf.summary.FileWriter(cfg.logs_path + 'train/', graph=graph)
        valid_writer = tf.summary.FileWriter(cfg.logs_path + 'valid/')
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
                    logging.info('epoch %s - step %s' % (epoch,step))
                    logging.info('runtime for batch of ' + str(batch_size * cfg.display_step) + ' ' + str(time.time()-current_time))
                    current_time = time.time()
                    c, acc, cm= sess.run([cost, accuracy,confusion_matrix], feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.keep_prob})

                    print(c, acc)
                    print(cm)
                    print(advanced_gen.batches_counter)

                if global_step % cfg.display_step_val == 0:
                    val_batch_gen = valid_corpus.batch_gen(len_valid, do_mfcc=True)
                    val_batch_x, val_batch_y = next(val_batch_gen)
                    summary_val, c_val, acc_val = sess.run([summaries, cost, accuracy],
                                                           feed_dict={x: val_batch_x, y: val_batch_y, keep_prob: 1})
                    valid_writer.add_summary(summary_val, global_step)
                    print("validation:", c_val, acc_val)

                step += 1
                global_step += 1
            # if epoch % cfg.epochs_per_save == 0:
            print('saving model...', end='')
            model_name = 'model_%s_bsize%s_e%s.ckpt' % ('mfcc',batch_size,epoch)

            s_path = saver.save(sess, cfg.logs_path + model_name)
            print("Model saved in file: %s" % s_path)


        print("Optimization Finished!")

def predict():
    fn_model = 'models/model0/model_mfcc_bsize256_e4.ckpt'
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
    #res = debug_model()
    train_model()

