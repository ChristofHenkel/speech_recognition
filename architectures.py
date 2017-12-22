import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn


class BaselineSilence:
    """
    Simple ConvNet with max Pooling
    """

    def __init__(self):
        pass

    def calc_logits(self,x,keep_prob,num_classes):


        x2 = x
        for i in [0, 1]:
            x2 = layers.conv2d(x2, num_outputs=  24 * 2**i, kernel_size=3, stride=1,
                               activation_fn=tf.nn.elu
                               )
            x2 = layers.max_pool2d(x2, 2, 2)

        # -> (512,1,1,32)
        x2 = tf.reduce_max(x2, axis=[1, 2], keep_dims=True)

        x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        x2 = layers.fully_connected(x2, 16, activation_fn=tf.nn.relu)
        x2 = layers.fully_connected(x2, 8, activation_fn=tf.nn.relu)
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        x2 = layers.fully_connected(x2, num_classes, activation_fn=tf.nn.relu)

        logits = tf.squeeze(x2, [1, 2])
        return logits

class BaselineSilence2:
    """
    Simple ConvNet with max Pooling
    """

    def __init__(self, cfg):
        pass

    def calc_logits(self,x,keep_prob,num_classes):


        x2 = x
        for i in [0,1]:
            x2 = layers.conv2d(x2, num_outputs=  16 * 2**i, kernel_size=(2,3), stride=1,
                               activation_fn=tf.nn.elu
                               )
            x2 = layers.max_pool2d(x2, kernel_size=(2,4), stride=2)
        print(x2)
        # -> (512,1,1,32)
        x2 = tf.reduce_max(x2, axis=[1, 2], keep_dims=True)
        #x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)
        x2 = layers.conv2d(x2, 16, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        x2 = layers.conv2d(x2, 16, 1, 1, activation_fn=tf.nn.elu)
        x2 = layers.fully_connected(x2, 16, activation_fn=tf.nn.relu)
        x2 = layers.fully_connected(x2, 8, activation_fn=tf.nn.relu)
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        x2 = layers.fully_connected(x2, num_classes, activation_fn=tf.nn.relu)

        logits = tf.squeeze(x2, [1, 2])
        return logits

class cnn_one_fpool3:
    """Builds a standard convolutional model.
    This is roughly the network labeled as 'cnn-trad-fpool3' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
    """
    def __init__(self,cfg):

        pass

    def calc_logits(self,x,keep_prob,num_classes):


        x2 = x
        x2 = layers.conv2d(x2, num_outputs=54, kernel_size=(4, 70), stride=1,activation_fn=tf.nn.elu)
        x2 = layers.max_pool2d(x2, kernel_size=(3, 1), stride=1)
        x2 = layers.conv2d(x2, num_outputs=54, kernel_size=(2, 35), stride=1,activation_fn=tf.nn.elu)
        x2 = layers.max_pool2d(x2, kernel_size=(2, 1), stride=1)


        # -> (512,1,1,32)
        x2 = tf.reduce_max(x2, axis=[1, 2], keep_dims=True)
        #x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)
        x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        #x2 = layers.conv2d(x2, 16, 1, 1, activation_fn=tf.nn.elu)
        x2 = layers.fully_connected(x2, 64, activation_fn=tf.nn.relu)
        x2 = layers.fully_connected(x2, 64, activation_fn=tf.nn.relu)
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        x2 = layers.fully_connected(x2, num_classes, activation_fn=tf.nn.relu)

        logits = tf.squeeze(x2, [1, 2])
        return logits

class cnn_one_fpool3_rnn:
    """Builds a standard convolutional model.
    This is roughly the network labeled as 'cnn-trad-fpool3' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
    """

    def __init__(self, cfg):
        pass

    def calc_logits(self, x, keep_prob, num_classes):
        x2 = x
        x2 = layers.conv2d(x2, num_outputs=54, kernel_size=(4, 70), stride=1, activation_fn=tf.nn.elu)
        x2 = layers.max_pool2d(x2, kernel_size=(3, 1), stride=1)
        x2 = layers.conv2d(x2, num_outputs=54, kernel_size=(2, 35), stride=1, activation_fn=tf.nn.elu)
        x2 = layers.max_pool2d(x2, kernel_size=(2, 1), stride=1)
        x2 = layers.conv2d(x2, num_outputs=54, kernel_size=(1, 20), stride=1, activation_fn=tf.nn.elu)
        x2 = layers.max_pool2d(x2, kernel_size=(2, 1), stride=1)

        x2 = x2[:,:,0,:]

        #nceps = 18
        #x2 = tf.unstack(x2, 12, 1)
        #x2 = tf.unstack(x2, 12, 1)


        # Define a lstm cell with tensorflow
        with tf.variable_scope('lstm1'):
            stacked_fw_rnn = []
            for fw_Lyr in range(2):
                fw_cell = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0, state_is_tuple=True)  # or True
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)
                stacked_fw_rnn.append(fw_cell)
            fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)
            #fw_reset_state = fw_multi_cell.zero_state(self.batch_params.batch_size, dtype=tf.float32)
            #fw_state = fw_reset_state

            output, _ = tf.nn.dynamic_rnn(fw_multi_cell, x2, dtype=tf.float32)
            # Select last output.
            output = tf.transpose(output, [1, 0, 2])
            fw_outputs = tf.gather(output, int(output.get_shape()[0]) - 1)
            #fw_outputs = tf.reshape(fw_output, [-1, 128])

        with tf.variable_scope('lstm2'):
            stacked_bw_rnn = []
            for bw_Lyr in range(2):
                bw_cell = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0, state_is_tuple=True)  # or True
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=keep_prob)
                stacked_bw_rnn.append(bw_cell)
            bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)
            #bw_reset_state = bw_multi_cell.zero_state(self.batch_params.batch_size, dtype=tf.float32)
            #bw_state = bw_reset_state

            bw_output, _ = tf.nn.dynamic_rnn(bw_multi_cell, x2, dtype=tf.float32)
            # Select last output.
            bw_output = tf.transpose(bw_output, [1, 0, 2])
            bw_outputs = tf.gather(bw_output, int(bw_output.get_shape()[0]) - 1)


        #outputs = fw_outputs[-1] + bw_outputs[-1]
        outputs = fw_outputs + bw_outputs
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        #x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)


        logits = layers.fully_connected(x3, num_classes, activation_fn=tf.nn.relu)
        return logits

class cnn_trad_fpool3:
    """Builds a standard convolutional model.
    This is roughly the network labeled as 'cnn-trad-fpool3' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
    """
    def __init__(self):
        pass

    def calc_logits(self,x,keep_prob,num_classes):


        x2 = x
        x2 = layers.conv2d(x2, num_outputs=64, kernel_size=(8, 60), stride=1,activation_fn=tf.nn.elu)
        x2 = layers.max_pool2d(x2, kernel_size=(3, 1), stride=1)
        x2 = layers.conv2d(x2, num_outputs=64, kernel_size=(4, 30), stride=1,activation_fn=tf.nn.elu)
        x2 = layers.max_pool2d(x2, kernel_size=(1, 1), stride=1)



        # -> (512,1,1,32)
        x2 = tf.reduce_max(x2, axis=[1, 2], keep_dims=True)
        #x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)
        x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        #x2 = layers.conv2d(x2, 16, 1, 1, activation_fn=tf.nn.elu)
        x2 = layers.fully_connected(x2, 128, activation_fn=tf.nn.relu)
        x2 = layers.fully_connected(x2, 128, activation_fn=tf.nn.relu)
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        x2 = layers.fully_connected(x2, num_classes, activation_fn=tf.nn.relu)

        logits = tf.squeeze(x2, [1, 2])
        return logits

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

    def __init__(self, cfg):
        self.test = ''
        self.config = cfg
        self.hparams = {'use_batch_norm':cfg.use_batch_norm,
                        'is_training':cfg.is_training}
        self.num_conv_layers = 3

    def calc_logits(self,x,keep_prob,num_classes):
        if self.hparams['use_batch_norm']:
            x2 = layers.batch_norm(x, is_training=self.hparams['is_training'])
        else:
            x2 = x
        for i in range(self.num_conv_layers):

            x2 = layers.conv2d(x2, 8 * 2 ** i, 3, 1,
                               activation_fn=tf.nn.elu,
                              normalizer_fn=layers.batch_norm if self.hparams['use_batch_norm'] else None,
                               normalizer_params={'is_training': self.hparams['is_training']}

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
            #fw_reset_state = fw_multi_cell.zero_state(self.batch_params.batch_size, dtype=tf.float32)
            #fw_state = fw_reset_state

            output, _ = tf.nn.dynamic_rnn(fw_multi_cell, x2, dtype=tf.float32)
            # Select last output.
            output = tf.transpose(output, [1, 0, 2])
            fw_outputs = tf.gather(output, int(output.get_shape()[0]) - 1)
            #fw_outputs = tf.reshape(fw_output, [-1, 128])

        with tf.variable_scope('lstm2'):
            stacked_bw_rnn = []
            for bw_Lyr in range(2):
                bw_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0, state_is_tuple=True)  # or True
                stacked_bw_rnn.append(bw_cell)
            bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)
            #bw_reset_state = bw_multi_cell.zero_state(self.batch_params.batch_size, dtype=tf.float32)
            #bw_state = bw_reset_state

            bw_output, _ = tf.nn.dynamic_rnn(bw_multi_cell, x2, dtype=tf.float32)
            # Select last output.
            bw_output = tf.transpose(bw_output, [1, 0, 2])
            bw_outputs = tf.gather(bw_output, int(bw_output.get_shape()[0]) - 1)


        #outputs = fw_outputs[-1] + bw_outputs[-1]
        outputs = fw_outputs + bw_outputs
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        #x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)


        logits = layers.fully_connected(x3, num_classes, activation_fn=tf.nn.relu)


        return logits

class Baseline8:
    """
    ConvNet with stacked RNN on top
    """

    def __init__(self, cfg):
        self.test = ''
        self.config = cfg
        self.hparams = {'use_batch_norm':cfg.use_batch_norm,
                        'is_training':cfg.is_training}
        self.num_conv_layers = 3

    def calc_logits(self,x,keep_prob,num_classes):
        if self.hparams['use_batch_norm']:
            x2 = layers.batch_norm(x, is_training=self.hparams['is_training'])
        else:
            x2 = x
        #regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        for i in range(self.num_conv_layers):

            x2 = layers.conv2d(x2, 8 * 2 ** i, 3, 1,
                               activation_fn=tf.nn.elu,
                              normalizer_fn=layers.batch_norm if self.hparams['use_batch_norm'] else None,
                               normalizer_params={'is_training': self.hparams['is_training']},
                               #weights_regularizer=regularizer
                               )
            #x2 = tf.layers.conv2d(x2, 8 * 2 ** i, 3, 1,kernel_regularizer=regularizer)
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
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)
                stacked_fw_rnn.append(fw_cell)
            fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)
            #fw_reset_state = fw_multi_cell.zero_state(self.batch_params.batch_size, dtype=tf.float32)
            #fw_state = fw_reset_state

            output, _ = tf.nn.dynamic_rnn(fw_multi_cell, x2, dtype=tf.float32)
            # Select last output.
            output = tf.transpose(output, [1, 0, 2])
            fw_outputs = tf.gather(output, int(output.get_shape()[0]) - 1)
            #fw_outputs = tf.reshape(fw_output, [-1, 128])

        with tf.variable_scope('lstm2'):
            stacked_bw_rnn = []
            for bw_Lyr in range(2):
                bw_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0, state_is_tuple=True)  # or True
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=keep_prob)
                stacked_bw_rnn.append(bw_cell)
            bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)
            #bw_reset_state = bw_multi_cell.zero_state(self.batch_params.batch_size, dtype=tf.float32)
            #bw_state = bw_reset_state

            bw_output, _ = tf.nn.dynamic_rnn(bw_multi_cell, x2, dtype=tf.float32)
            # Select last output.
            bw_output = tf.transpose(bw_output, [1, 0, 2])
            bw_outputs = tf.gather(bw_output, int(bw_output.get_shape()[0]) - 1)


        #outputs = fw_outputs[-1] + bw_outputs[-1]
        outputs = fw_outputs + bw_outputs
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        #x2 = layers.conv2d(x2, 32, 1, 1, activation_fn=tf.nn.elu)   # we can use conv2d 1x1 instead of dense
        x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)


        logits = layers.fully_connected(x3, num_classes, activation_fn=tf.nn.relu)


        return logits

