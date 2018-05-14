import tensorflow as tf
import models
import layers.layers as layers

class TFBSFirstModel(models.Model):
    def __init__(self):
        models.Model.__init__(self)

    def _predict(self, input_batch, batch_size):

        tf_channels, genome_channels, labels, bio_sample_one_hot_ = input_batch
        with tf.name_scope('main_model'):
            genome_conv_stage = [
                layers.Simple1DCNN([3,6,32], [32], 'relu'),
                layers.Simple1DCNN([3,32,32], [32], 'relu'),
                layers.SimplePool([2],[2]),
                layers.Simple1DCNN([3,32,32],[32], 'relu')
            ]

            amino_acid_conv_stage = [
                layers.Simple1DCNN([3,21,32],[32],'relu'),
                layers.Simple1DCNN([3,32,64],[64], 'relu'),
                layers.SimplePool([2], [2]),
                layers.Simple1DCNN([3,64,64], [64], 'relu')
            ]

            genome_conv_output = models.Model._get_layer_output(self, genome_conv_stage, genome_channels)
            amino_acid_output = models.Model._get_layer_output(self, amino_acid_conv_stage, tf_channels)

            genome_dense_input = tf.reshape(genome_conv_output, [batch_size, -1])
            amino_acid_dense_input = tf.reshape(amino_acid_output, [batch_size, -1])

            #print genome_dense_input, amino_acid_dense_input

            fc_input = tf.concat([genome_dense_input, amino_acid_dense_input, bio_sample_one_hot_], 1)
            fc_input_size = fc_input.get_shape().as_list()[1]

            #print fc_input

            fc_layer = layers.SimpleDNN(fc_input_size,1024,'relu', 0.75)

            fc_hidden_output = fc_layer.forward(fc_input)

            #print fc_hidden_output


            model_logit_output = tf.nn.softmax(layers.SimpleDNN(1024, 3, 'relu').forward(fc_hidden_output))

            #print model_logit_output

            model_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(model_logit_output, 1))

            model_accuracy = tf.reduce_mean(tf.cast(model_prediction, tf.float32))

            return model_logit_output, model_prediction, model_accuracy

    def _train(self, input_batch, batch_size):
        #with self.graph.as_default():
        _, _, labels, _ = input_batch

        with tf.name_scope('training'):
            model_output, _, model_accuracy = TFBSFirstModel._predict(self, input_batch, batch_size)


            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=model_output))

            train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            return train_op, cross_entropy, model_accuracy
