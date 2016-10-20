
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


def input():
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


def _activation_summary(x):
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name+"/activations",x)
    tf.scalar_summary(tensor_name+"/sparsity",tf.nn.zero_fraction(x))



