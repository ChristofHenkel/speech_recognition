from batch_gen import BatchGen
import tensorflow as tf

class Config:
    soundcorpus_fp = 'assets/corpora/corpus1/validation.soundcorpus.part1.p'
    batch_size = 1000
    size = 16000

cfg = Config()
gen = BatchGen(batch_size = cfg.batch_size,soundcorpus_fp = cfg.soundcorpus_fp)

# x,y = next(gen.batch_gen())


# Define Graph

size = cfg.size
batch_size = cfg.batch_size

graph = tf.Graph()
with graph.as_default():
    # tf Graph input
    with tf.name_scope("Input"):

        x = tf.placeholder(tf.float32, shape=(None, size), name="input")
        x.set_shape([batch_size, size])
        y = tf.placeholder(tf.int64, shape=(None, size), name="input")
        # keep_prob = tf.placeholder(tf.float32, name="dropout")


    
