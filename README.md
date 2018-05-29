# genomics

# How to RUN:
1. change line 222 in https://github.com/rohit06nitbpl/genomics/blob/master/tfbs/source_code/tensorflow/experiments/dreamc/data_processing.py  according to location of data_dir on your disk
2. Run python file https://github.com/rohit06nitbpl/genomics/blob/master/tfbs/source_code/tensorflow/experiments/dreamc/experiments.py without argument on latest tensorflow environment.

Device Placement and Training log are done.

# How to RUN Tensorboard:

tensorboard --logdir=genomics/tfbs/source_code/tensorflow/dataset/sample_dataset/logs

Graph and Scaler can be visualised in Tensorboard
 
# Description
I used this code in very initial experiments, I also used Amino Acid sequences of TF as 
additional feature, just to see its usefulness in the prediction of TF binding 
even for unknown TF. Our earlier focus was to predict TF binding sites for even 
for unknown TFs (i.e. not seen in experiments), but later on, I only focused on improving Dream Challenge results.
 
I am reading tsv format files in this code rather than zipped version. Although, lately I have used zipped versions and tools like bedtool etc. mentioned on website.
 
I am creating input Tensors on the fly. This way, I am not storing huge one hot matrix in memory or disk (data set is huge in genomics).
 
Earlier, I was reading file in many threads and filling three queues for Ambigous(A), Bound(B), Unbound(U) samples. 
Then I was using a random_shuffle_queue to sample equal number of samples and to form final batch (balanced classes in each batch) for training.
 
But due to unbalanced number of three classes (A, B, U), training had to wait for say(B) samples to be found deep in the files using queue pipeline.
 
Therefore, I pre-processed files to segregate A, B and U sample in file to three separate files. Now, as in this code, I am creating one thread per 
class (A, B, U). Each thread quickly find sample for each class and enqueque it in queue pipeline. 
 
It has been been suggested in TensorFlow to use large file instead of many small file, cause it will bottleneck systems file handling resources.
 
In fact, In sample data set, I used in this code, size of class A file is very small, which makes file to be closed and reopened again very quickly for different epochs. 
This makes it slow to fill queue for class A. If you run my code, you will see, training waits intermittently for queue of class A. 
 
It can be made much faster on cluster using big data technologies like Apache Spark and Hadoop, and They can gather data and push into queue much quickly 
than single system. 
 
# Main features of the code:
1. Very modular code, This same code can be used for training various kind of data and model by writing new stuff using inheritance in object oriented manner.
2. For Example, for sample data set, I have only specific code in TFBSAAFileReader, TFBS_AA_CNN_MODEL class and data_processing.py (file to process data itself).
3. Every other piece if code in pipeline can be leveraged as it is.
4. It will be easy to extend this code on multi-gpu, modular manner would make it easier to do device placement of nodes. 
5. I have used proper name scoping in again in modular manner to visualise graph in Tensorboard, along with tf.summaries,  
6. data is passed using data_dict rather than command line parameter, cause this dict can be stored and retrieved in automated manner.
 
# To Dos
1. Finish multi GPU model using Tower Fashion
2. Finish Parameter Visualization, Embedding visualization
3. Complete tf.data.Dataset and compare performances
4. Complete Distributed implementation using https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks 
5. Watch out https://www.tensorflow.org/versions/r1.1/performance/performance_models
