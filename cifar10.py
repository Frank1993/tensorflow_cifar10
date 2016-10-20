
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

    with tf.variable_scope("conv1") as scope:
        kernel = tf.get_variable("weights",shape =[5,5,3,64] )




