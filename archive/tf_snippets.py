import tensorflow as tf
from tensorflow.contrib import signal
import numpy as np


def preprocess(x):
    specgram = signal.stft(
        x,
        400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        160,  # 16000 * 0.010 -- default stride

    )
    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))
    x2 = tf.stack([amp, phase], axis=3)  # shape is [bs, time, freq_bins, 2]
    x2 = tf.to_float(x2)
    return x2

#probs = tf.nn.softmax(logits2)
#pred = tf.argmax(logits2, 1)
#if cfg.lr_decay != None:
#    learning_rate = tf.train.exponential_decay(cfg.learning_rate, iteration,
#                                               100000, cfg.lr_decay, staircase=True)