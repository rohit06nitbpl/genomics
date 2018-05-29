# Genomics

# Modular Distributed TensorFlow Framework

# Extract Transform Load Pipeline
1. Extract: Read using local file system (HDD or SSD) or remote file system (GCS or HDFS)
2. Transform: Effectively utilize CPU cores to parse and perform pre processing, batching
3. Load: Heavy lifting of computation on many GPUS or TPUs locally or across cluster

# Feeding Data to Graph 
1. Initialize Tensors with input data into the Graph: Bloat Graph size, Used for trivial problem and on single GPU, Very inefficient to duplicate Graph on multiple devices
2. Feed data into Graph using dictionary: Huge memory utilization, and also huge disk requirement for huge preprocessed data
3. Input pipeline using Queue: Queue Runner are implemented in python, Efficient but can not saturate current generation multiple GPUs
4. Input pipeline using tf.Data() API: Implemented using C++, parallelize I/O, transform and load steps using background threads, Recommended

# Variable Distribution in Multi GPU and Distributed Model
1. Parameter Servers: Parameters are pinned to parameter server, and they are implicitly copied to worker, gradient is computed at worker and aggregated at parameter server
2. Replicated Variables: Each GPUs or worker has its own copy of variable, single device (CPU or GPU) is then used to aggregate gradient
3. Replicated Variables in Distributed Environment: Each worker has local copy of variables, local copy is then updated using parameter server which aggregates gradient 

Keeping local copies of variables allows for faster computation

# Code Description
1. Multi GPUs Model, Uses Input Pipeline using Queue, Variable distribution are done using Parameter Server approach
2. Parameter are pinned to CPU, and GPUs if available serves as worker
3. Very Modular and Object Oriented Design, Core module abstract away basic routine functionality and also provide layers to implement new models
4. For example, for sample dataset, I only implemented TFBSAAFileReader, TFBS_AA_CNN_MODEL classes apart from pre-processing(initial, data specific)
5. proper name scoping for visualization of Graph in Tensorborad along with tf.summaries
6. data is passed using data_dict rather than command line parameter, cause this dict can be stored and retrieved in automated manner

# How to RUN : Sample Dataset
1. Change line 222 in https://github.com/rohit06nitbpl/genomics/blob/master/tfbs/source_code/tensorflow/models/dreamc/pre_processing.py#L222  according to location of data_dir on your disk
2. Run python file https://github.com/rohit06nitbpl/genomics/blob/master/tfbs/source_code/tensorflow/experiments/dreamc/experiments.py without argument on latest tensorflow environment.
3. Add available GPUs on line 19 in https://github.com/rohit06nitbpl/genomics/blob/master/tfbs/source_code/tensorflow/models/dreamc/experiments.py#L19

Device Placement and Training log are done.

# How to RUN Tensorboard : Sample Dataset

tensorboard --logdir=genomics/tfbs/source_code/tensorflow/dataset/dreamc/sample_dataset/attempt0/train/logs

Graph and Scaler can be visualised in Tensorboard
 
# Sample Data Description
It is small self made data in similar format as DREAM-ENCODE TF in-vivo binding challenge data

I used this model in this code as very initial experiments, I also used Amino Acid sequences of TF as 
additional feature, just to see its usefulness in the prediction of TF binding 
even for unknown TF (i.e. for which experiments are not done). Our earlier focus was to predict TF binding sites for even 
for unknown TFs , but later on, I only focused on improving Dream Challenge results.
 
I am reading tsv format files in this code rather than zipped version. Although, lately I have used zipped versions and tools like bedtool etc. mentioned on DREAM-ENCODE website.
 
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
than a single system. 
 
# TODO
1. Upload 1D implementation of Capsules, I used for TF Binding Problem
2. Add Pipeline using tf.data() API and compare performances using real world data like mnist
3. Complete Distributed implementation using https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks 

