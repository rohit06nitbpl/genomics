import tensorflow as tf
import time
import pipeline.data_feeder as data_feeder

class Model:
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(log_device_placement=True))
        self.coord = tf.train.Coordinator()

    def _get_layer_output(self, layers_list, input_data):
        print input_data
        output = input_data
        index = 1
        for layer in layers_list:
            print index, layer
            output = layer.forward(output)
            print output
        return output

    def load_data(self, data_dict, batch_size, n_epoch, feeder_type):
        if feeder_type == 'queue':
            with tf.name_scope('queue_feeder'):
                return data_feeder.QueueFeeder(self.sess, self.coord, data_dict).next_batch(batch_size, n_epoch)
        elif feeder_type == 'dataset':
            raise NotImplementedError('Not implemented')
        else:
            raise NotImplementedError('Not implemented')

    def train(self, data_dict, batch_size, n_epoch, max_step, feeder_type):
        with self.graph.as_default():
            data_dir = data_dict['data_dir']
            save_after_steps = data_dict['save_after_steps']
            summary_after_steps = data_dict['summary_after_steps']

            input_batch, threads_list, curr_queue_size_op_list = Model.load_data(self, data_dict,batch_size,n_epoch,feeder_type)

            train_op, cost, accuracy = self._train(input_batch, batch_size)

            init_op = tf.global_variables_initializer()

            tf.summary.scalar("cost", cost)
            tf.summary.scalar("accuracy", accuracy)
            summary_op = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(data_dir+'/logs',self.graph)
            saver = tf.train.Saver()

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()


            print 'Creating Session...'
            # Create a session for running operations in the Graph.
            with self.sess as sess: # running default graph in this session

                # Initialize the variables (like the epoch counter).
                sess.run(init_op)
                print 'Variable Initialized...'
                #enqueue_threads = [ qr.create_threads(sess, coord=coord, start=True) for qr in queue_runner_list]
                #threads = tf.train.start_queue_runners(coord=coord)

                try:
                    start_time = time.clock()
                    summary_step_time = time.clock()
                    n_steps = 1
                    wait_time = 0
                    sleep_time = 10 #seconds
                    while not self.coord.should_stop():
                        queue_sizes = [sess.run(op) for op in curr_queue_size_op_list]

                        if n_steps <= max_step or max_step == -1:
                            # Run training steps
                            while  min(queue_sizes) <= batch_size: # batch_size is arbitrary in this line, but it's ok
                                if self.coord.should_stop():
                                    break
                                print 'Steps Done:', n_steps-1, ', Waiting for queue to fill...', \
                                    'Queue 1:', queue_sizes[0], ', Queue 2:', queue_sizes[1], ', Queue 3:', queue_sizes[2]
                                time.sleep(sleep_time)
                                wait_time = wait_time + sleep_time
                                queue_sizes = [sess.run(op) for op in curr_queue_size_op_list]

                            if n_steps%summary_after_steps == 0:
                                result = sess.run([train_op, summary_op], options=run_options, run_metadata=run_metadata)
                                #summary_writer.add_run_metadata(run_metadata,n_steps)
                                summary_writer.add_summary(result[1], n_steps)

                                print 'Batch Done:', str(n_steps), ', Time Taken:', str(time.clock() - summary_step_time), \
                                    ', Total Time Taken:', str(time.clock() - start_time), ', Total Wait Time:', str(wait_time)

                                summary_step_time = time.clock()
                            else:
                                sess.run([train_op])

                            if n_steps%save_after_steps == 0:
                                saver.save(sess, data_dir + '/saved_model/train_chk_pt', global_step=n_steps)

                            n_steps = n_steps + 1


                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                finally:
                    # When done, ask the threads to stop.
                    self.coord.request_stop()

                # Wait for threads to finish.
                self.coord.join(threads_list)
                #sess.close()
                print 'Total Training Time Taken: '+ str(time.clock() - start_time)
