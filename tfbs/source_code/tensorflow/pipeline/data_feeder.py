import tensorflow as tf
import threading
import time

import tfbs.source_code.tensorflow.models.dreamc.data_reader


class Feeder:
    def __init__(self):
        pass

class QueueFeeder(Feeder):

    safety_margin = 1

    def __init__(self, sess, coord, data_dict):
        Feeder.__init__(self)
        self.sess = sess
        self.coord = coord
        self.data_dict = data_dict

        self.train_info = {}
        self.test_info = {}

    def next_batch(self, batch_size, n_epoch, dataset_type):

        max_queue_size_per_class = self.data_dict['max_queue_size_per_class']
        n_classes = self.data_dict['n_classes']
        min_after_dequeue = batch_size*5 # TODO 5 is arbitrary

        input_per_class_list = []
        enqueue_threads_list = []
        queue_curr_size_op_list = []

        with tf.name_scope('shuffle_queue'):
            for class_type in self.data_dict['ground_truth'].keys():
                with tf.name_scope('class_queue'):
                    queue = tf.FIFOQueue(max_queue_size_per_class, dtypes=[tf.float32, tf.float32, tf.float32, tf.float32])
                    reader_per_class = tfbs.source_code.tensorflow.models.dreamc.data_reader.TFBSAAFileReader(self.sess, self.coord, self.data_dict, class_type)
                    queue_pl = reader_per_class.get_queue_placeholders()
                    enqueue_op = queue.enqueue(queue_pl)

                    enqueue_threads_list.append(threading.Thread(target=reader_per_class.read, args=(queue_pl, enqueue_op, n_epoch, dataset_type)))

                    tf.summary.scalar(class_type+":curr_queue_size", queue.size())
                    queue_curr_size_op_list.append(queue.size())

                    input_per_class_list.append(queue.dequeue())

            capacity = n_classes*max_queue_size_per_class + batch_size + self.safety_margin
            input_batch = tf.train.shuffle_batch_join(
                input_per_class_list, batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue, shapes=reader_per_class.get_queue_ele_shape())

            #for t in enqueue_threads_list:
                #t.start()

            other_threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            #time.sleep(60)  # Allowed 60 seconds initial time for Extract and Transform step on data
            #print 'Allowed 60 seconds initial time for Extract and Transform step on data...'

            #threads_list =  enqueue_threads_list + other_threads

            if dataset_type == 'train':
                self.train_info['other_threads_list'] = other_threads
                self.train_info['enqueue_threads_list'] = enqueue_threads_list
                self.train_info['queue_curr_size_op_list'] = queue_curr_size_op_list

            elif dataset_type == 'test':
                self.test_info['other_threads_list'] = other_threads
                self.train_info['enqueue_threads_list'] = enqueue_threads_list
                self.test_info['queue_curr_size_op_list'] = queue_curr_size_op_list
            else:
                raise NameError('Unknown dataset type')

            return input_batch

    def get_other_thread_list(self, dataset_type):
        if dataset_type == 'train':
            return self.train_info['other_threads_list']
        elif dataset_type == 'test':
            return self.test_info['other_threads_list']
        else:
            raise NameError('Unknown dataset type')

    def get_enqueue_thread_list(self, dataset_type):
        if dataset_type == 'train':
            return self.train_info['enqueue_threads_list']
        elif dataset_type == 'test':
            return self.test_info['enqueue_threads_list']
        else:
            raise NameError('Unknown dataset type')

    def get_queue_curr_size_op_list(self, dataset_type):
        if dataset_type == 'train':
            return self.train_info['queue_curr_size_op_list']
        elif dataset_type == 'test':
            return self.test_info['queue_curr_size_op_list']
        else:
            raise NameError('Unknown dataset type')


    '''
    def next_batch(self, data_dict, class_batch_size, n_epochs = 1, n_file_queue_reader = 1):
        capacity = self.buffer_elements + (class_batch_size + self.safety_margin) * n_file_queue_reader
        input_batch_per_class_list = []
        input_batch = []
        for class_type in data_dict['ground_truth'].keys():
            filename_queue = tf.train.string_input_producer(data_dict['ground_truth'][class_type],
                                                            num_epochs=n_epochs, shuffle=True)
            samples_list_per_class = [data_reader.TFBSAAFileReader(filename_queue, data_dict)
                                     for _ in range(n_file_queue_reader)]

            input_batch_per_class = tf.train.shuffle_batch_join(
                samples_list_per_class, batch_size=class_batch_size, capacity=capacity,
                min_after_dequeue=self.buffer_elements)
            input_batch_per_class_list.append(input_batch_per_class)

        if input_batch_per_class_list:
            input_batch_rank = len(input_batch_per_class_list[0].get_shape().as_list())
            for i in range(input_batch_rank):
                temp = []
                for j in range(input_batch_per_class_list):
                    temp.append(input_batch_per_class_list[j][i])
                element = tf.concat(temp, axis=0)
                input_batch.append(element)

        return input_batch
    '''


class DatasetFeeder(Feeder):
    def __init__(self):
        pass
