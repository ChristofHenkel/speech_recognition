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


def RNN(x, weights, biases, timesteps, num_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def stacked_BI_RNN(x,batch_size,num_layers,keep_prob,n_hidden):


    with tf.name_scope("Embedding"):
        embedding = tf.get_variable("embedding", [vocabulary_size, embed_size], dtype=tf.float32)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")
        # Creates Tensor with TensorShape([Dimension(batchsize), Dimension(nsteps), Dimension(embed_size)])
    # fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)  # or True
    # if keep_probability < 1:
    #    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)

    # bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # if keep_probability < 1:
    #    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=keep_prob)

    stacked_fw_rnn = []
    for fw_Lyr in range(num_layers):
        fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)  # or True
        if keep_prob < 1:
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)
        stacked_fw_rnn.append(fw_cell)
    fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

    stacked_bw_rnn = []
    for bw_Lyr in range(num_layers):
        bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)  # or True
        if keep_prob < 1:
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=keep_prob)
        stacked_bw_rnn.append(bw_cell)
    bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

    # fw_multi_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * layers, state_is_tuple=True)  # or True
    # bw_multi_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * layers, state_is_tuple=True)

    fw_reset_state = fw_multi_cell.zero_state(batch_size, dtype=tf.float32)
    bw_reset_state = bw_multi_cell.zero_state(batch_size, dtype=tf.float32)

    fw_state = fw_reset_state
    bw_state = bw_reset_state

    # TODO test implementation of

    fw_outputs = []
    with tf.variable_scope("RNN_fw_output"):
        for time_step in range(seq_len - 1):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            inputs = embedded_input[:, time_step, :]
            (cell_output, fw_state) = fw_multi_cell(inputs, fw_state)
            fw_outputs.append(cell_output)

    fw_final_state = fw_state
    fw_output = tf.reshape(tf.concat(axis=1, values=fw_outputs), [-1, n_hidden])

    bw_outputs = []
    with tf.variable_scope("RNN_bw_output"):
        for time_step in reversed(range(1, seq_len)):
            if time_step < seq_len - 1: tf.get_variable_scope().reuse_variables()
            inputs = embedded_input[:, time_step, :]
            (cell_output, bw_state) = bw_multi_cell(inputs, bw_state)
            bw_outputs.append(cell_output)

    bw_final_state = bw_state
    bw_outputs = bw_outputs[::-1]
    bw_output = tf.reshape(tf.concat(axis=1, values=bw_outputs), [-1, n_hidden])

    softmax_w_fw = tf.get_variable("softmax_w_fw", [n_hidden, vocabulary_size], dtype=tf.float32)
    softmax_b_fw = tf.get_variable("softmax_b_fw", [vocabulary_size], dtype=tf.float32)
    logits_fw = tf.matmul(fw_output, softmax_w_fw) + softmax_b_fw

    softmax_w_bw = tf.get_variable("softmax_w_bw", [n_hidden, vocabulary_size], dtype=tf.float32)
    softmax_b_bw = tf.get_variable("softmax_b_bw", [vocabulary_size], dtype=tf.float32)
    logits_bw = tf.matmul(bw_output, softmax_w_bw) + softmax_b_bw

    # Add zero logits for first entry of sequence
    # unflatten logits
    logits_fw = tf.reshape(logits_fw, [batch_size, seq_len - 1, vocabulary_size])
    # add zeros
    z = tf.Variable(tf.zeros([batch_size, 1, vocabulary_size]), dtype=tf.float32)
    logits_fw = tf.concat([z, logits_fw], 1)
    # flatten
    logits_fw = tf.reshape(logits_fw, [-1, vocabulary_size])

    logits_bw = tf.reshape(logits_bw, [batch_size, seq_len - 1, vocabulary_size])
    # add zeros
    z = tf.Variable(tf.zeros([batch_size, 1, vocabulary_size]), dtype=tf.float32)
    logits_bw = tf.concat([logits_bw, z], 1)
    # flatten
    logits_bw = tf.reshape(logits_bw, [-1, vocabulary_size])

    if not cfg.use_fw_bw_aggregation_layer:
        logits = logits_fw + logits_bw
    else:
        # TODOS 54 should be vocab size
        logits = fulconn_layer(tf.concat([logits_fw, logits_bw], 1), 54)

