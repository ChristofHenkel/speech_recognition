import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

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


class Model1:


    def __init__(self, cfg):
        self.test = ''
        self.config = cfg
        self.hparams = {'use_batch_norm':True,
                        'is_training':True}

    def calc_logits(self,x,keep_prob,num_classes):

        x2 = x[:,:,:,0]
        x2 = layers.batch_norm(x2, is_training=self.hparams['is_training'])
        for i in [0, 1, 2]:
            x2 = layers.conv2d(x2, 8 * 2**i, 3, 1,
                               activation_fn=tf.nn.elu,
                               normalizer_fn=layers.batch_norm if self.hparams['use_batch_norm'] else None,
                               normalizer_params={'is_training': self.hparams['is_training']}
                               )
            #x2 = layers.max_pool2d(x2, 2, 2)
            x2 = tf.layers.max_pooling1d(x2, 2, 2)

        x2 = tf.reduce_max(x2, axis=[1, 2], keep_dims=True)

        x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        x2 = layers.fully_connected(x2, 16, activation_fn=tf.nn.relu)
        x2 = layers.fully_connected(x2, num_classes, activation_fn=tf.nn.relu)
        logits = tf.squeeze(x2, [1])

        return logits

class Model2:
    """
    Simple ConvNet with max Pooling
    """

    def __init__(self, cfg):
        self.test = ''
        self.config = cfg
        self.hparams = {'use_batch_norm':cfg.use_batch_norm,
                        'is_training':cfg.is_training}

    def calc_logits(self,x,keep_prob,num_classes):

        if self.hparams['use_batch_norm']:
            x2 = layers.batch_norm(x, is_training=self.hparams['is_training'])
        else:
            x2 = x
        for i in [0, 1]:
            x2 = layers.conv2d(x2, num_outputs=  8 * 2**i, kernel_size=3, stride=1,
                               activation_fn=tf.nn.elu,
                               normalizer_fn=layers.batch_norm if self.hparams['use_batch_norm'] else None,
                               normalizer_params={'is_training': self.hparams['is_training']}
                               )
            x2 = layers.max_pool2d(x2, 2, 2)

        # -> (512,1,1,32)
        x2 = tf.reduce_max(x2, axis=[1, 2], keep_dims=True)

        x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        x2 = layers.fully_connected(x2, 16, activation_fn=tf.nn.relu)
        x2 = layers.fully_connected(x2, num_classes, activation_fn=tf.nn.relu)

        logits = tf.squeeze(x2, [1, 2])
        return logits




class Model3:

    def __init__(self, cfg):
        self.test = ''
        self.config = cfg
        self.hparams = {'use_batch_norm':True,
                        'is_training':True}

    def calc_logits(self,x,keep_prob,num_classes):



        x2 = layers.batch_norm(x, is_training=self.hparams['is_training'])

        x2 = layers.conv3d(x2, 8, 3, 1,
                           activation_fn=tf.nn.elu,
                           normalizer_fn=layers.batch_norm if self.hparams['use_batch_norm'] else None,
                           normalizer_params={'is_training': self.hparams['is_training']}
                           )
        if self.hparams['is_training']:
            x2 = tf.nn.dropout(x2, keep_prob=keep_prob)

        x2 = layers.max_pool2d(x2, 2, 2)

        x2 = layers.conv3d(x2, 16, 3, 1,
                           activation_fn=tf.nn.elu,
                           normalizer_fn=layers.batch_norm if self.hparams['use_batch_norm'] else None,
                           normalizer_params={'is_training': self.hparams['is_training']}
                           )

        x2 = layers.max_pool2d(x2, 2, 2)

        x2 = layers.conv3d(x2, 32, 3, 1,
                           activation_fn=tf.nn.elu,
                           normalizer_fn=layers.batch_norm if self.hparams['use_batch_norm'] else None,
                           normalizer_params={'is_training': self.hparams['is_training']}
                           )

        x2 = layers.avg_pool2d(x2, 2, 2)

        portion_m = 0.7
        mpool = tf.reduce_max(x2, axis=[1, 2], keep_dims=True)
        apool = tf.reduce_mean(x2, axis=[1, 2], keep_dims=True)

        x2 = portion_m * mpool + (1-portion_m) * apool
        # we can use conv2d 1x1 instead of dense

        # (128, 1, 1, 32) -> (128, 1, 1, 32)
        x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)

        x2 = layers.fully_connected(x2, 16, activation_fn=tf.nn.relu)
        # again conv2d 1x1 instead of dense layer
        # (128, 1, 1, 32) -> (128, 1, 1, 12)
        # x2 = layers.conv2d(x2, num_classes, 1, 1, activation_fn=None)
        x2 = layers.fully_connected(x2, num_classes, activation_fn=tf.nn.relu)

        # -> (128, 1, 1, 12) - > (128, 12)
        logits = tf.squeeze(x2, [1, 2])
        return logits

class Model4:
    """
    test of RNN
    """

    def __init__(self, cfg):
        self.test = ''
        self.config = cfg
        self.hparams = {'use_batch_norm':True,
                        'is_training':True}

    # input (batch_size, 99, 13)
    # Conv
    # RNN


    def calc_logits(self,x,keep_prob,num_classes):


        x2 = x[:,:,:,0]
        #x2 = tf.transpose(x2,perm=[0, 2, 1])
        x2 = tf.unstack(x2, 99, 1)


        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x2, dtype=tf.float32)
        #x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        x3 = layers.fully_connected(outputs[-1], 16, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(x3, num_classes, activation_fn=tf.nn.relu)

        #logits = tf.squeeze(x2, [1, 2])
        return logits

class Model5:
    """
    ConvNet with RNN on top
    """

    def __init__(self, cfg):
        self.test = ''
        self.config = cfg
        self.hparams = {'use_batch_norm':cfg.use_batch_norm,
                        'is_training':cfg.is_training}
        self.num_conv_layers = 3

    def calc_logits(self,x,keep_prob,num_classes):
        # was still wrong in model44
        if self.hparams['use_batch_norm']:
            x2 = layers.batch_norm(x, is_training=self.hparams['is_training'])
        else:
            x2 = x

        for i in range(self.num_conv_layers):
            x2 = layers.conv2d(x2, 8 * 2 ** i, 3, 1,
                               activation_fn=tf.nn.elu,
                               normalizer_fn=layers.batch_norm if self.hparams['use_batch_norm'] else None,
                               normalizer_params={'is_training': self.hparams['is_training']},
                               #weights_initializer=layers.xavier_initializer(uniform=False),
                               # biases_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)

                               )
            x2 = layers.max_pool2d(x2, 2, 2)

        x2 = x2[:,:,0,:]

        #nceps = 18
        x2 = tf.unstack(x2, 12, 1)
        #x2 = tf.unstack(x2, 12, 1)


        # Define a lstm cell with tensorflow
        with tf.variable_scope('lstm1'):
            fw_lstm_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)
            fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)
            fw_outputs, fw_states = rnn.static_rnn(fw_lstm_cell, x2, dtype=tf.float32)

        with tf.variable_scope('lstm2'):
            bw_lstm_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)
            bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)
            bw_outputs, fw_states = rnn.static_rnn(bw_lstm_cell, x2, dtype=tf.float32)
        # Get lstm cell outpu

        outputs = fw_outputs[-1] + bw_outputs[-1]
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        #x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
        #x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        #x3 = layers.fully_connected(outputs, 16, activation_fn=tf.nn.relu)

        logits = layers.fully_connected(x3, num_classes, activation_fn=tf.nn.relu)


        return logits

class Model6:
    """
    flat ConvNet with max Pooling
    """

    def __init__(self, cfg):
        self.test = ''
        self.config = cfg
        self.hparams = {'use_batch_norm': True,
                        'is_training': True}

    def calc_logits(self, x, keep_prob, num_classes):
        x2 = layers.batch_norm(x, is_training=self.hparams['is_training'])
        for _ in [0, 1, 2]:
            x2 = layers.conv2d_in_plane(x2, kernel_size=3, stride=1,
                               activation_fn=tf.nn.elu,
                               normalizer_fn=layers.batch_norm if self.hparams['use_batch_norm'] else None,
                               normalizer_params={'is_training': self.hparams['is_training']}
                               )
            x2 = layers.max_pool2d(x2, 2, 2)
            #x2 = tf.nn.dropout(x2, keep_prob=keep_prob)

        x2 = tf.reduce_max(x2, axis=[1, 2], keep_dims=True)

        x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)  # we can use conv2d 1x1 instead of dense
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        x2 = layers.fully_connected(x2, 16, activation_fn=tf.nn.relu)
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        x2 = layers.fully_connected(x2, num_classes, activation_fn=tf.nn.relu)

        logits = tf.squeeze(x2, [1, 2])
        return logits

class Baseline7:
    """
    ConvNet with stacked RNN on top
    """

    def __init__(self, cfg, batch_params):
        self.test = ''
        self.config = cfg
        self.batch_params = batch_params
        self.hparams = {'use_batch_norm':cfg.use_batch_norm,
                        'is_training':cfg.is_training}
        self.num_conv_layers = 3

    def calc_logits(self,x,keep_prob,num_classes):
        # was still wrong in model44
        if self.hparams['use_batch_norm']:
            x2 = layers.batch_norm(x, is_training=self.hparams['is_training'])
        else:
            x2 = x

        for i in range(self.num_conv_layers):
            x2 = layers.conv2d(x2, 8 * 2 ** i, 3, 1,
                               activation_fn=tf.nn.elu,
                               normalizer_fn=layers.batch_norm if self.hparams['use_batch_norm'] else None,
                               normalizer_params={'is_training': self.hparams['is_training']},
                               #weights_initializer=layers.xavier_initializer(uniform=False),
                               # biases_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)

                               )
            x2 = layers.max_pool2d(x2, 2, 2)

        x2 = x2[:,:,0,:]

        #nceps = 18
        #x2 = tf.unstack(x2, 12, 1)
        #x2 = tf.unstack(x2, 12, 1)


        # Define a lstm cell with tensorflow
        with tf.variable_scope('lstm1'):
            stacked_fw_rnn = []
            for fw_Lyr in range(2):
                fw_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0, state_is_tuple=True)  # or True
                stacked_fw_rnn.append(fw_cell)
            fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)
            fw_reset_state = fw_multi_cell.zero_state(self.batch_params.batch_size, dtype=tf.float32)
            fw_state = fw_reset_state

            fw_output, fw_state = fw_multi_cell(x2, fw_state)
            fw_outputs = tf.reshape(fw_output, [-1, 128])

        with tf.variable_scope('lstm2'):
            stacked_bw_rnn = []
            for bw_Lyr in range(2):
                bw_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0, state_is_tuple=True)  # or True
                stacked_bw_rnn.append(bw_cell)
            bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)
            bw_reset_state = bw_multi_cell.zero_state(self.batch_params.batch_size, dtype=tf.float32)
            bw_state = bw_reset_state

            bw_output, bw_state = bw_multi_cell(x2, bw_state)
            bw_outputs = tf.reshape(bw_output, [-1, 128])


        outputs = fw_outputs[-1] + bw_outputs[-1]
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        #x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
        #x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        #x3 = layers.fully_connected(outputs, 16, activation_fn=tf.nn.relu)

        logits = layers.fully_connected(x3, num_classes, activation_fn=tf.nn.relu)


        return logits



