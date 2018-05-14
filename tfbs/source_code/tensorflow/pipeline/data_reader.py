import tensorflow as tf
import numpy as np
import time

class DataReader:
    def __init__(self):
        pass

class TFDataReader(DataReader):
    def __init__(self, sess, coord, data_dict, class_type):
        DataReader.__init__(self)
        self.data_dir = data_dict['data_dir']
        self.n_classes = data_dict['n_classes']
        self.flanking_region = data_dict['flanking_region']
        self.class_type = class_type
        self.bio_tf_train = data_dict['bio_tf_train']
        self.bio_tf_test = data_dict['bio_tf_test']
        self.tf_aa_dict = data_dict['tf_aa_dict']
        self.bio_samples = data_dict['bio_samples']
        self.ground_truth = data_dict['ground_truth'][class_type]
        self.test_data = data_dict['test_data'][class_type]
        self.class_index = data_dict['class_type_index_dict'][class_type]
        self.nucleotide_dict = data_dict['nucleotide_dict']
        self.sess = sess
        self.coord = coord



    def read(self, queue_pl, enqueue_op, n_epochs, dataset = 'train', log_data = False):
        if log_data: f_data_log = open(self.data_dir+'/data_log.txt', mode='a')

        #local_sess = tf.Session()
        #buffer = []
        coord_stop_requested = False

        if dataset == 'train':
            input_files = self.ground_truth
        else:
            input_files = self.test_data

        aa_one_hot_size = 21
        max_aa_size = 0
        for value in self.tf_aa_dict.values():
            max_aa_size = max(max_aa_size, len(value))

        genome_npy_dict = {}
        genome_one_hot_size = len(self.nucleotide_dict)
        labels = [0.0, 0.0, 0.0]
        labels[self.class_index] = 1.0

        dnase_dict = {}
        extra_genome_channel = 1 #DNASE

        for epoch in range(n_epochs):
            #print 'epoch', epoch
            if coord_stop_requested:
                break
            file_process_start_time = time.clock()
            for input_file in input_files:
                if coord_stop_requested:
                    break
                #print 'input_file', input_file
                input_file_token = input_file.split('.')
                bio_sample = input_file_token[-3]
                transFactor = input_file_token[-4].split('/')[-1]
                tf_aa_seq = self.tf_aa_dict[transFactor]

                bio_sample_one_hot = [0.0 for x in range(len(self.bio_samples))]
                bio_sample_one_hot[self.bio_samples.index(bio_sample)] = 1.0

                aa_one_hot_matrix = [[0.0 for x in range(aa_one_hot_size)] for y in range(max_aa_size)]
                for i in range(len(tf_aa_seq)):
                    aa_one_hot_matrix[i][tf_aa_seq[i]] = 1.0

                with open(input_file) as f:
                    line = f.readline().split('\n')[0]
                    #print 'line', line
                    while line != '' and not coord_stop_requested:
                        chrm,start,stop,label = line.split('\t')
                        start = int(start)
                        stop = int(stop)

                        genome_start = start - self.flanking_region
                        genome_stop = stop + self.flanking_region
                        seq_len = genome_stop - genome_start


                        if not genome_npy_dict.has_key(chrm): genome_npy_dict[chrm] = \
                            np.load(self.data_dir + '/processed_data/genome_npy/hg19.' + chrm + '.npy', mmap_mode='r')

                        if not dnase_dict.has_key(bio_sample): dnase_dict[bio_sample] = \
                            np.load(self.data_dir + '/processed_data/dnase_npy/' + bio_sample
                                    + '.' + chrm + '.npy', mmap_mode='r')[genome_start:genome_stop]

                        # queue size sanity check ??

                        genome_seq = genome_npy_dict[chrm][genome_start:genome_stop]

                        if len(genome_seq) == seq_len :
                            genome_one_hot_matrix = [[0.0 for x in range(genome_one_hot_size+extra_genome_channel)]
                                                     for y in range(seq_len)]

                            for i in range(seq_len):
                                genome_one_hot_matrix[i][genome_seq[i]] = 1.0
                                if i < len(dnase_dict[bio_sample]):
                                    genome_one_hot_matrix[i][5] = dnase_dict[bio_sample][i]
                                else:
                                    genome_one_hot_matrix[i][5] = 0.0

                            label_one_hot = [0.0 for x in range(self.n_classes)]
                            label_one_hot[self.class_index] = 1.0

                            if not self.coord.should_stop():
                                self.sess.run(enqueue_op, {queue_pl[0]: aa_one_hot_matrix, queue_pl[1]: genome_one_hot_matrix,
                                                           queue_pl[2]: label_one_hot, queue_pl[3]: bio_sample_one_hot})
                                #print epoch, n_epochs, chrm
                            else:
                                print 'coordinator stop requested...\n'
                                coord_stop_requested = True

                        line = f.readline().split('\n')[0]
                        if log_data:
                            f_data_log.writelines(transFactor + '\n')
                            f_data_log.writelines(bio_sample + '\n')
                            f_data_log.writelines('\t'.join(str(j) for j in i) + '\n' for i in aa_one_hot_matrix)
                            f_data_log.writelines('\t'.join(str(j) for j in i) + '\n' for i in genome_one_hot_matrix)
                            f_data_log.writelines('\t'.join(str(i) for i in bio_sample_one_hot) + '\n')
                            f_data_log.writelines('\t'.join(str(i) for i in label_one_hot) + '\n')
                            f_data_log.flush()

                #print 'processed: ' + input_file + 'Time: ' + str(time.clock() - file_process_start_time)
                if log_data:
                    f_data_log.writelines('processed: ' + input_file + 'Time: ' + str(time.clock() - file_process_start_time) + '\n')
                    f_data_log.flush()
                    file_process_start_time = time.clock()
            #print 'Class:', self.class_index, ', Epoch Read:' + str(epoch)
            if log_data:
                f_data_log.writelines('epoch ' + str(epoch) + '\n')
                f_data_log.flush()
        if log_data:
            f_data_log.close()
        #local_sess.close()
        if not coord_stop_requested:
            print 'All epochs have been read for Class:', self.class_type
            self.coord.request_stop()

    def get_queue_placeholders(self):
        aa_one_hot_size = 21
        max_aa_size = 0
        for value in self.tf_aa_dict.values():
            max_aa_size = max(max_aa_size, len(value))

        seq_len = 2*self.flanking_region+200
        genome_channels = 6

        aa_one_hot_matrix_pl = tf.placeholder(tf.float32, shape=[max_aa_size, aa_one_hot_size])
        gdata_onehot_pl = tf.placeholder(tf.float32, shape=[seq_len, genome_channels])
        label_one_hot_pl = tf.placeholder(tf.float32, shape=[self.n_classes])
        bio_sample_one_hot = tf.placeholder(tf.float32, shape=[len(self.bio_samples)])

        return [aa_one_hot_matrix_pl, gdata_onehot_pl, label_one_hot_pl, bio_sample_one_hot]

    def get_queue_ele_shape(self):
        aa_one_hot_size = 21
        max_aa_size = 0
        for value in self.tf_aa_dict.values():
            max_aa_size = max(max_aa_size, len(value))

        seq_len = 2 * self.flanking_region + 200
        genome_channels = 6

        return [[max_aa_size, aa_one_hot_size], [seq_len, genome_channels], [self.n_classes], [len(self.bio_samples)]]

    '''
    def read(self, filename_queue_per_class, data_dict):
        n_classes = data_dict['n_classes']
        data_dir = data_dict['data_dir']
        flanking_region = data_dict['flanking_region']
        seq_len = flanking_region*2 + 200
        genome_one_hot_size = 6
        tf_aa_dict = data_dict['tf_aa_dict']
        bio_sample_size = len(data_dict['bio_samples'])
        max_aa_size = 0
        aa_one_hot_size = 21
        for value in tf_aa_dict.values():
            max_aa_size = max(max_aa_size, len(value))

        tf_reader = tf.TextLineReader(skip_header_lines=1)
        key, value = tf_reader.read(filename_queue_per_class)

        print key, value

        key_split = key.split('.')

        class_type = key_split[-2]
        bio_sample = key_split[-3]
        transFactor = key_split[-4]


        #record_defaults = tf.constant(1.0, shape=[[max_aa_size, aa_one_hot_size],[seq_len, genome_one_hot_size],[n_classes],[bio_sample_size]])

        record_defaults = ['chr',0,0,'U']

        chr, start, end, _class_type = tf.decode_csv(value, record_defaults=record_defaults, field_delim="\t")

        g_np_array = np.load(data_dir + 'processed_data/genome_npy/hg19.' + chr + '.npy', mmap_mode='r')
    '''



