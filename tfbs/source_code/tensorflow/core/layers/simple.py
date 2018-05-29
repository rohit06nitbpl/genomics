import tensorflow as tf

import variables

class Simple1DCNN:
    def __init__(self, kernel_shape, bias_shape, var_scope, activation_fn = None,
                 stride = 1, padding = 'VALID', name = None):

        if name:
            self.name = name
        else:
            self.name = "simple_1d_cnn"

        self.activation_fn = activation_fn
        self.stride = stride
        self.padding = padding

        with tf.variable_scope(var_scope) as scope:
            self.kernel = variables.Weights('/cpu:0').get('kernel', kernel_shape)
            self.bias = variables.Biases('/cpu:0').get('bias', bias_shape)

    def forward(self, input_data):
        with tf.name_scope(self.name):
            if self.activation_fn:
                if self.activation_fn == 'relu':
                    conv_1d = tf.nn.conv1d(input_data, self.kernel, stride=self.stride, padding=self.padding) + self.bias
                    return tf.nn.relu(conv_1d)
                else:
                    raise NotImplementedError('Not implemented')
            else:
                return tf.nn.conv1d(input_data, self.kernel, stride=self.stride,
                                    padding=self.padding) + self.bias


class SimpleDNN:
    def __init__(self, len_input_vector, n_units, var_scope, activation_fn = None,
                 keep_prob = 1.0, name = None):

        if name:
            self.name = name
        else:
            self.name = "simple_dnn"

        self.activation_fn = activation_fn
        self.keep_prob = keep_prob

        with tf.variable_scope(var_scope) as scope:
            self.weight = variables.Weights('/cpu:0').get('fc_weights', [len_input_vector, n_units])
            self.bias = variables.Biases('/cpu:0').get('bias', [n_units])

    def forward(self, input_data):
        with tf.name_scope(self.name):
            matrix_product = tf.matmul(input_data, self.weight) + self.bias
            if self.activation_fn:
                if self.activation_fn == 'relu':
                    activation = tf.nn.relu(matrix_product)
                else:
                    raise NotImplementedError('Not implemented')
            else:
                activation = matrix_product #Just Matrix Multiplication

            if self.keep_prob < 1.0:
                return tf.nn.dropout(activation, self.keep_prob)
            else:
                return activation


class SimplePool:
    def __init__(self, window_shape, strides, pooling_type='MAX', padding='VALID',
                 name = None):
        if name:
            self.name = name
        else:
            self.name = "simple_pool"

        self.window_shape = window_shape
        self.strides = strides
        self.pooling_type = pooling_type
        self.padding = padding

    def forward(self, input_data):
        with tf.name_scope(self.name):
            return tf.nn.pool(input_data, window_shape=self.window_shape,
                          pooling_type=self.pooling_type, padding=self.padding,
                          strides=self.strides)
