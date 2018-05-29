import tensorflow as tf

class Variables:
    def __init__(self, device_setter):
        self.device_setter = device_setter

    def _create_var_summary(self, var, verbose):
        if verbose:
            with tf.name_scope('var_summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)

                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
        else:
            pass

class Weights(Variables):
    def __init__(self, device_setter=None):
        Variables.__init__(self, device_setter)

    def get(self, name, shape, stddev=0.1, verbose=False):

        with tf.device(self.device_setter):
            with tf.name_scope(name):
                weights = tf.get_variable(name,
                                          shape,
                                          initializer=tf.truncated_normal_initializer(stddev=stddev))

        self._create_var_summary(weights, verbose)
        return weights

class Biases(Variables):
    def __init__(self, device_setter=None):
        Variables.__init__(self, device_setter)

    def get(self, name, shape, value=0.1, verbose=False):

        with tf.device(self.device_setter):
            with tf.name_scope(name):
                biases = tf.get_variable(name,
                                          shape,
                                          initializer=tf.constant_initializer(value))

        self._create_var_summary(biases, verbose)
        return biases