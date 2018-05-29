import sys
import os
import tensorflow as tf

from pre_processing import sample_data_dict
from tfbs_aa_cnn_model import TFBS_AA_CNN_MODEL

#import tfbs.source_code.tensorflow.core.layers.variables as variables

def test():
    data_dict = sample_data_dict()
    hparams = tf.contrib.training.HParams(
        decay_rate=0.96,
        decay_steps=2000,
        learning_rate=0.001,
    )
    model  = TFBS_AA_CNN_MODEL(hparams, 'queue')
    gpus = []
    #gpus = ['/gpu:0', '/gpu:1'] #Assuming gpu 1 and gpu 2 are available
    model.train(data_dict, gpus, total_batch_size=10, n_epoch=1000, max_steps=-1)



if __name__ == "__main__":
    if len(sys.argv) == 1:
        #print 'CUDA_VISIBLE_DEVICES', os.environ["CUDA_VISIBLE_DEVICES"]
        '''
        tower = []
        tower_var = []
        sess = tf.Session()

        with tf.device('/cpu:0'):

            queue = tf.FIFOQueue(5, dtypes=[tf.float32])
            enqueue_op1 = queue.enqueue(tf.constant([1.0]))
            enqueue_op2 = queue.enqueue(tf.constant([0.0]))


        for device in ['/gpu:0', '/gpu:1']:
            with tf.device(device):
                device_name = device.replace('/', '_')
                device_name = device_name.replace(':', '_')
                with tf.name_scope('tower_%s' % device_name) as scope:
                    var = variables.Weights('/cpu:0').get('w_1', [1])

                    with tf.device('/cpu:0'):
                        x = queue.dequeue()

                    tower_op = var*x
                    tower_var.append(var)
                    tower.append(tower_op)
                    tf.get_variable_scope().reuse_variables()


        output = tower[0] + tower[1]
        output = output/2.0


        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(enqueue_op1)
        sess.run(enqueue_op2)
        print 'w_1:', sess.run(tower_var[0])
        print 'w_1:', sess.run(tower_var[1])
        print 'output:', sess.run(output)

        print 'done'
        '''
        test()
    else:
        print 'Error! Many arguements!'



