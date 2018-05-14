import tensorflow as tf

class Simple1DCNN:
    def __init__(self, kernel_shape, bias_shape, activation_fn = None,
                 stride = 1, padding = 'VALID', name = None):
        if name:
            self.name = name
        else:
            self.name = "simple_1d_cnn"

        self.activation_fn = activation_fn
        self.stride = stride
        self.padding = padding
        with tf.name_scope(self.name):
            self.kernel = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1))
            self.bias = tf.Variable(tf.constant(0.1, shape=bias_shape))

    def forward(self, input_data):
        with tf.name_scope(self.name):
            if self.activation_fn:
                if self.activation_fn == 'relu':
                    return tf.nn.relu(tf.nn.conv1d(input_data, self.kernel, stride=self.stride,
                                           padding=self.padding) + self.bias)
                else:
                    raise NotImplementedError('Not implemented')


class SimpleDNN:
    def __init__(self, len_input_vector_, n_units, activation_fn = None,
                 keep_prob = 1.0, name = None):
        if name:
            self.name = name
        else:
            self.name = "simple_dnn"

        self.activation_fn = activation_fn
        self.keep_prob = keep_prob

        with tf.name_scope(self.name):
            self.weight = tf.Variable(tf.random_normal([len_input_vector_, n_units], stddev=0.1))
            self.bias = tf.Variable(tf.constant(0.1, shape=[n_units]))

    def forward(self, input_data):
        with tf.name_scope(self.name):
            matrix_product = tf.matmul(input_data, self.weight) + self.bias
            if self.activation_fn:
                if self.activation_fn == 'relu':
                    activation = tf.nn.relu(matrix_product)
                else:
                    raise NotImplementedError('Not implemented')

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
