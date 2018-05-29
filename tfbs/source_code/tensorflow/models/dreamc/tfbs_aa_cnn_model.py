import tensorflow as tf
import time
from tfbs.source_code.tensorflow.core.abstract_model import Model
import tfbs.source_code.tensorflow.core.layers.simple as simple

class TFBS_AA_CNN_MODEL(Model):
    def __init__(self, hparams, feeder_type):
        Model.__init__(self, hparams, feeder_type)

    def _forward(self, input_batch, batch_size):
        tf_channels, genome_channels, labels, bio_sample_one_hot_ = input_batch

        with tf.name_scope('main_model'):
            #CNNS
            genome_conv_stage = [
                simple.Simple1DCNN([3, 6, 32], [32], 'genome_conv1', 'relu'),
                simple.Simple1DCNN([3, 32, 32], [32], 'genome_conv2', 'relu'),
                simple.SimplePool([2], [2]),
                simple.Simple1DCNN([3, 32, 32], [32], 'genome_conv3', 'relu')
            ]

            amino_acid_conv_stage = [
                simple.Simple1DCNN([3, 21, 32], [32], 'aa_conv1', 'relu'),
                simple.Simple1DCNN([3, 32, 64], [64], 'aa_conv2', 'relu'),
                simple.SimplePool([2], [2]),
                simple.Simple1DCNN([3, 64, 64], [64], 'aa_conv3', 'relu')
            ]

            #LAYER OUTPUT
            genome_conv_output = Model._get_layers_output(self, genome_conv_stage, genome_channels)
            amino_acid_output = Model._get_layers_output(self, amino_acid_conv_stage, tf_channels)

            #RESHAPES
            genome_dense_input = tf.reshape(genome_conv_output, [batch_size, -1])
            amino_acid_dense_input = tf.reshape(amino_acid_output, [batch_size, -1])

            #CONCATS
            fc_input = tf.concat([genome_dense_input, amino_acid_dense_input, bio_sample_one_hot_], 1)
            fc_input_size = fc_input.get_shape().as_list()[1]

            #FCs
            fc_stage = [
                simple.SimpleDNN(fc_input_size, 1024, 'hidden_dnn', 'relu', 0.75),
                simple.SimpleDNN(1024, 3, 'output_dnn', 'relu')
            ]

            #LAYER OUTPUT
            fc_output = Model._get_layers_output(self, fc_stage, fc_input)

            #SOFTMAX
            model_logit_output = tf.nn.softmax(fc_output)

        return model_logit_output

    def _predict(self, model_output, input_batch, batch_size):
        #_, _, labels, _ = input_batch
        with tf.name_scope('prediction'):
            model_prediction = tf.argmax(model_output, 1)
        return  model_prediction

    def _eval(self, model_prediction, input_batch, batch_size):
        _, _, labels, _ = input_batch
        with tf.name_scope('eval'):
            model_prediction_match = tf.equal(tf.argmax(labels, 1), model_prediction)
            model_accuracy = tf.reduce_mean(tf.cast(model_prediction_match, tf.float32))
        return model_accuracy

    def _loss(self, model_output, input_batch, batch_size):
        _, _, labels, _ = input_batch
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=model_output))
        return cross_entropy

    def _run_train(self, sess, summary_writer, last_step, max_steps, summary_after_steps, start_time):

        train_op = self.total_batch_train_op
        summary_op = self.summary_op

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary_step_time = time.time()

        for index in range(last_step, max_steps):
            curr_step = index + 1
            if curr_step % summary_after_steps == 0:
                result = sess.run([train_op, summary_op], options=run_options, run_metadata=run_metadata)
                # summary_writer.add_run_metadata(run_metadata,n_steps)
                summary_writer.add_summary(result[1], curr_step)

                print 'Steps Done:', str(curr_step), ', Time Taken:', str(time.time() - summary_step_time), \
                    ', Total Time Taken:', str(time.time() - start_time)

                summary_step_time = time.time()
            else:
                sess.run([train_op])

    def _make_single_tower(self, input_batch, batch_size_per_tower, device):
        with tf.device(device):
            device_name = device.replace('/','_')
            device_name = device_name.replace(':','_')
            with tf.name_scope('tower_%s' % device_name) as scope:
                model_output = self._forward(input_batch, batch_size_per_tower)
                model_prediction = self._predict(model_output, input_batch, batch_size_per_tower)
                model_accuracy = self._eval(model_prediction, input_batch, batch_size_per_tower)
                model_loss = self._loss(model_output, input_batch, batch_size_per_tower)

                tf.get_variable_scope().reuse_variables()
                tower_grad = self.optimizer.compute_gradients(model_loss)
        return model_prediction, model_accuracy, tower_grad, model_loss

    def _make_towers(self, input_batch, batch_size_per_tower, gpus):

        with tf.variable_scope(tf.get_variable_scope()):
            model_predictions = []
            model_accuracies = []
            tower_grads = []
            model_losses = []
            if gpus:
                for gpu in gpus:
                    single_tower_output = self._make_single_tower(input_batch, batch_size_per_tower, gpu)
                    model_predictions.append(single_tower_output[0])
                    model_accuracies.append(single_tower_output[1])
                    tower_grads.append(single_tower_output[2])
                    model_losses.append(single_tower_output[3])
            else:
                single_tower_output = self._make_single_tower(input_batch, batch_size_per_tower, '/cpu:0')
                model_predictions.append(single_tower_output[0])
                model_accuracies.append(single_tower_output[1])
                tower_grads.append(single_tower_output[2])
                model_losses.append(single_tower_output[3])

        self._aggregate_towers((model_predictions, model_accuracies, tower_grads, model_losses))

    def _aggregate_towers(self, all_tower_output):

        model_predictions, model_accuracies, tower_grads, model_losses = all_tower_output

        grads = self._average_gradients(tower_grads)
        self.total_batch_train_op = self.optimizer.apply_gradients(grads,
                                                                   global_step=self.global_step)
        stacked_accuracy = tf.stack(model_accuracies)
        self.total_batch_accuracy = tf.reduce_mean(stacked_accuracy)

        self.total_batch_prediction = tf.concat(model_predictions, 0)

        tf.summary.scalar("total_batch_loss", tf.reduce_sum(model_losses))
        tf.summary.scalar("total_batch_accuracy", self.total_batch_accuracy)

        self.summary_op = tf.summary.merge_all()
