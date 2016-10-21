
"""
build the cifar10 gragh
"""

from __future__ import  absolute_import
from __future__ import  division
from __future__ import  print_function

import os
import  sys
import tarfile
import gzip
import  tensorflow as tf
import  urllib


from cifar10_input import  *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_dir",'tmp/cifar10_data',"path to the cifar10 dataset directory")
tf.app.flags.DEFINE_integer("batch_size",200, "number of examples used in one batch")

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def maybe_download_and_extract():
    dest_directory = FLAGS.data_dir

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = DATA_URL.split('/')[-1]

    filepath = os.path.join(dest_directory,filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))

            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)

        statinfo = os.stat(filepath)
        print("Successfully downloaded",filename,statinfo.st_size,"bytes")

        tarfile.open(filepath,'r:gz').extractall(dest_directory)

def inputs():
    if not FLAGS.data_dir:
        raise ValueError("Please supply a data dir")

    # get the dir of extracted bin file of cifar10
    data_dir = os.path.join(FLAGS.data_dir,"cifar-10-batches-bin")

    images,labels = input_pipeline(data_dir,FLAGS.batch_size)

    return images,labels

def inference(images):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # conv1
    with tf.variable_scope("conv1") as scope:
        kernel = tf.get_variable("weights",shape =[5,5,3,64],initializer = tf.truncated_normal_initializer(1))
        conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding = "SAME")
        biases = tf.get_variable("biases",[64],tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name = scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1,ksize = [1,3,3,1],strides = [1,2,2,1], padding = "SAME",name = "pool1")

    # normal
    norm1 = tf.nn.lrn(pool1,4,bias = 1.0,alpha = 0.001/9.0, beta = 0.75)

    # conv2
    with tf.variable_scope("conv2") as scope:
        kernel = tf.get_variable("weights", shape = [5,5,64,64], initializer = tf.truncated_normal_initializer(5e-2))

        conv = tf.nn.conv2d(norm1,kernel,[1,1,1,1], padding = "SAME")
        biases = tf.get_variable("biases",[64],tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name = scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2,4,bias = 1.0, alpha = 0.001/9.0, beta = 0.75, name = "norm2")

    # pool2
    pool2 = tf.nn.max_pool(norm2,ksize = [1,3,3,1], strides = [1,2,2,1], padding = "SAME", name = "pool2")

    #local3

    with tf.variable_scope("local3") as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2,[FLAGS.batch_size,-1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weight",shape = [dim,384], initializer = tf.truncated_normal_initializer(1e-2))

        biases = tf.get_variable("biases",[384],tf.constant_initializer(0.1))

        local3 = tf.nn.relu(tf.matmul(reshape,weights) + biases,name = scope.name)

    with tf.variable_scope("local4") as scope:
        weights = tf.get_variable("weight", shape = [384,192],initializer = tf.truncated_normal_initializer(1e-2))

        biases = tf.get_variable("biases",[192],tf.constant_initializer(0.1))

        local4 = tf.nn.relu(tf.matmul(local3,weights) + biases, name = scope.name)

    # softmax softmax(WX+b)
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weight", [192,NUM_CLASSES], initializer = tf.truncated_normal_initializer(1e-2))

        biases = tf.get_variable("biases", [NUM_CLASSES],tf.constant_initializer(0.0))

        softmax_linear = tf.add(tf.matmul(local4,weights), biases, name = scope.name)

        return softmax_linear


def loss(logits,labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels, name = "cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = "cross_entropy")
    return cross_entropy_mean


def train(loss,global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size

    decay_steps = int(num_batches_per_epoch* NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,decay_steps,LEARNING_RATE_DECAY_FACTOR, staircase = true)

    optimizer = tf.train.AdamOptimizer(lr)

    minimize_op=optimizer.minimize(loss,global_step)

    # Track the moving average of all trainable variable
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    variable_averages_op = variable_averages.apply(tf.traiable_variables())

    with tf.control_dependencies([minimize_op,variable_averages_op]):
        train_op = tf.no_op(name = "train")

    return train_op

def _activation_summary(x):
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name+"/activations",x)
    tf.scalar_summary(tensor_name+"/sparsity",tf.nn.zero_fraction(x))



