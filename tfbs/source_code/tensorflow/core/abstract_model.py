import tensorflow as tf
import os
import sys
import time
import tfbs.source_code.tensorflow.pipeline.data_feeder as data_feeder
#import tfbs.source_code.tensorflow.models.dreamc.data_reader as data_reeder

class Model:
    def __init__(self, hparams, feeder_type):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(log_device_placement=False))
        self.coord = tf.train.Coordinator()

        self.feeder_type = feeder_type

        self.hparams = hparams

        with self.graph.as_default():
            with tf.device('/cpu:0'):
                self.global_step = tf.get_variable(
                    'global_step', [],
                    initializer=tf.constant_initializer(0),
                    trainable=False)

                learning_rate = tf.train.exponential_decay(
                    learning_rate=hparams.learning_rate,
                    global_step=self.global_step,
                    decay_steps=hparams.decay_steps,
                    decay_rate=hparams.decay_rate)

                learning_rate = tf.maximum(learning_rate, 1e-6)

                self.optimizer = tf.train.AdamOptimizer(learning_rate)


    def _get_layers_output(self, layers_list, input_data):
        #print input_data
        output = input_data
        index = 1
        for layer in layers_list:
            #print index, layer
            output = layer.forward(output)
            #print output
        return output

    def _load_data(self, data_dict, batch_size, n_epoch, dataset_type):
        if self.feeder_type == 'queue':
            with tf.name_scope('queue_feeder'):
                self.data_feeder = data_feeder.QueueFeeder(self.sess, self.coord, data_dict)
                input_batch = self.data_feeder.next_batch(batch_size, n_epoch, dataset_type)
                print 'Queue Feeder created... '
                return input_batch
        elif self.feeder_type == 'dataset':
            raise NotImplementedError('Not implemented')
        else:
            raise NotImplementedError('Not implemented')

    def _forward(self, input_batch, batch_size):
        raise NotImplementedError('Not implemented')

    def _predict(self, model_output, input_batch, batch_size):
        raise NotImplementedError('Not implemented')

    def _eval(self, model_prediction, input_batch, batch_size):
        raise NotImplementedError('Not implemented')

    def _loss(self, model_output, input_batch, batch_size):
        raise NotImplementedError('Not implemented')

    def _run_predict(self, sess, ops):
        raise NotImplementedError('Not implemented')

    def _run_eval(self, sess, ops, writer, last_step, max_steps):
        raise NotImplementedError('Not implemented')

    def _run_train(self, sess, summary_writer, last_step, max_steps, summary_after_steps, start_time):
        raise NotImplementedError('Not implemented')

    def _make_single_tower(self, input_batch, batch_size_per_tower, device):
        raise NotImplementedError('Not implemented')

    def _make_towers(self, input_batch, batch_size_per_tower, gpus):
        raise NotImplementedError('Not implemented')

    def _aggregate_towers(self, all_tower_output):
        raise NotImplementedError('Not implemented')

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grads_and_vars in zip(*tower_grads):
            grads = tf.stack([g for g, _ in grads_and_vars])
            grad = tf.reduce_mean(grads, 0)

            v = grads_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _get_prev_step(self, path):
        file_name = os.path.basename(path)
        return int(file_name.split('-')[-1])

    def _load_best_trained_model(self, sess, saver, ckpt_dir):
        if tf.gfile.Exists(ckpt_dir):
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                prev_step = self._get_prev_step(ckpt.model_checkpoint_path)
            else:
                tf.gfile.DeleteRecursively(ckpt_dir)
                tf.gfile.MakeDirs(ckpt_dir)
                prev_step = 0
        else:
            tf.gfile.MakeDirs(ckpt_dir)
            prev_step = 0
        print 'Loaded best trained model...'
        return prev_step

    def _load_model(self, sess, saver, ckpt_url):
        saver.restore(sess, ckpt_url)
        model = self._get_prev_step(ckpt_url)
        print 'Loaded specific model...'
        return model

    def _run_sess(self, sess, summary_writer, last_step, max_steps, saver, save_after_steps, summary_after_steps,
                  start_time, train_dir, batch_size_per_tower, n_workers, dataset_type):

        actual_total_batch_size = batch_size_per_tower*n_workers

        steps_done = last_step
        load_time = 0
        wait_time = 0
        sleep_time = 10  # seconds
        first_load = True

        if self.feeder_type == 'queue':
            print 'feeder_type == queue'
            time.sleep(5)

            enqueue_threads_list =  self.data_feeder.get_enqueue_thread_list(dataset_type)
            for t in enqueue_threads_list:
                t.start()

            data_queue_curr_size_op_list = self.data_feeder.get_queue_curr_size_op_list(dataset_type)
            n_data_queue = len(data_queue_curr_size_op_list)
            min_data_queue_curr_size_for_actual_total_batch = actual_total_batch_size/n_data_queue + 1 # Added 1 to Ceil
            min_data_queue_curr_size_for_save_steps = save_after_steps*min_data_queue_curr_size_for_actual_total_batch

            print 'min_data_queue_curr_size_for_save_steps', min_data_queue_curr_size_for_save_steps

            coordinator_stop_signal = False
            should_stop = False

            while not should_stop:
                if (steps_done + save_after_steps) <= max_steps or max_steps == -1:
                    queue_sizes = [sess.run(op) for op in data_queue_curr_size_op_list]
                    while min(queue_sizes) <= min_data_queue_curr_size_for_save_steps and not coordinator_stop_signal:
                        if self.coord.should_stop():
                            coordinator_stop_signal = True

                        if first_load:
                            print 'Steps Done:', steps_done, ', Allowing initial load...', \
                                'Queue 1:', queue_sizes[0], ', Queue 2:', queue_sizes[1], ', Queue 3:', queue_sizes[2]
                            time.sleep(sleep_time)
                            load_time = load_time + sleep_time

                        else:
                            print 'Steps Done:', steps_done, ', Waiting for queue to fill...', \
                                'Queue 1:', queue_sizes[0], ', Queue 2:', queue_sizes[1], ', Queue 3:', queue_sizes[2]
                            time.sleep(sleep_time)
                            wait_time = wait_time + sleep_time

                        queue_sizes = [sess.run(op) for op in data_queue_curr_size_op_list]

                    first_load = False

                    if coordinator_stop_signal and min(queue_sizes) <= min_data_queue_curr_size_for_save_steps:
                        should_stop = True
                        print 'Data Exhausted, Current queue sizes:', 'Queue 1:', queue_sizes[0], ', Queue 2:', queue_sizes[1], ', Queue 3:', queue_sizes[2]

                        print 'Initial Load Time:', load_time, ', Total Wait Time:', wait_time

                    if min(queue_sizes) >= min_data_queue_curr_size_for_save_steps:
                        if dataset_type == 'train':
                            self._run_train(sess, summary_writer, steps_done,
                                            steps_done + save_after_steps, summary_after_steps, start_time)

                            steps_done = steps_done + save_after_steps
                            saver.save(sess, train_dir + '/ckpt/model.ckpt', global_step=steps_done)
                        elif dataset_type == 'test':
                            raise NotImplementedError('Not implemented')
                        else:
                            raise NotImplementedError('Not implemented')

            print 'Actual Total Steps Done:', steps_done

        elif self.feeder_type == 'dataset':
            raise NotImplementedError('Not implemented')
        else:
            raise NotImplementedError('Not implemented')

    def train(self, data_dict, gpus, total_batch_size, n_epoch=1, max_steps=-1):
        print 'Initializing training...'

        save_after_steps = data_dict['save_after_steps']
        summary_after_steps = data_dict['summary_after_steps']
        attempt_dir = data_dict['attempt_dir']
        train_dir = attempt_dir + '/train'

        #DECIDING BATCH SIZE PER WORKER
        if gpus:
            n_workers = len(gpus)
            batch_size_per_tower = total_batch_size / n_workers
        else:
            batch_size_per_tower = total_batch_size
            n_workers = 1

        print 'Given Total Batch Size:', total_batch_size, ', Actual Total Batch Size:', batch_size_per_tower * n_workers
        print 'Batch size per worker:', batch_size_per_tower, ', n_worker:', n_workers

        with self.graph.as_default():
            #DATA PIPELINE
            input_batch = Model._load_data(self, data_dict, batch_size_per_tower, n_epoch, 'train')
            time.sleep(5)
            '''
            reader_per_class = data_reeder.TFBSAAFileReader(self.sess, self.coord, data_dict, 'A')
            input_batch_placeholder = reader_per_class.get_queue_placeholders()
            input_batch_placeholder = [tf.stack([t,t,t,t]) for t in input_batch_placeholder] #assuming batch per tower = 4
            '''
            #PLACING OPS USING DATA PIPELINE
            self._make_towers(input_batch, batch_size_per_tower, gpus)
            print 'OPS placed and Towers created...'

            #PRINTING MODEL
            param_stats = tf.contrib.tfprof.model_analyzer.\
                print_model_analysis(tf.get_default_graph(),
                                     tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
            sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

            #INIT OP
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            #SUMMARY WRITER AND SAVER
            summary_writer = tf.summary.FileWriter(train_dir + '/logs', self.graph)
            saver = tf.train.Saver()

            with self.sess as sess:  # Running default graph in this session
                # Initialize the variables (like the epoch counter).
                sess.run(init_op)

                #LOAD MODEL IF SAVED PREVIOUSLY
                last_step = self._load_best_trained_model(sess, saver, train_dir+'/ckpt')

                print 'Initialization done...'
                start_time = time.time()
                try:
                    print 'Starting Training...'
                    self._run_sess(sess, summary_writer, last_step,
                                   max_steps, saver, save_after_steps, summary_after_steps,
                                   start_time, train_dir, batch_size_per_tower, n_workers, 'train')
                    print 'Ending Training...'

                except tf.errors.OutOfRangeError:
                    print('Training Exception -- Epoch limit reached')
                finally:
                    # When done, ask the threads to stop.
                    self.coord.request_stop()
                    summary_writer.close()
                    print 'Total Training Time Taken: ' + str(time.time() - start_time)

                # Wait for threads to finish.
                self.coord.join(self.data_feeder.get_other_thread_list(dataset_type='train'))


    def eval(self):
        pass

    def predict(self):
        pass