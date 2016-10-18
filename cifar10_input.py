"""Routine for decoding the cifar-10 binary files

the cifar10 dataset has 10 different categories and each category is in one file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

# the original image of cifar10 is 32*32, but here we use 24*24
IMAGAGE_SIZE = 24

NUM_CLASSES = 10

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 10000

def read_cifar10(filename_queue):
    #theCIFAR - 10 dataset的文件格式定义是：每条记录的长度都是固定的，一个字节的标签，后面是3072字节的图像数据。

    label_bytes = 1

    image_height = 32
    image_width = 32
    image_depth = 3
    image_bytes = image_height + image_width + image_depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes)

    key,value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    label = tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.int32)

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [image_depth, image_height, image_width])

    uint8_image = tf.transpose(depth_major,[1,2,0])

    image = tf.cast(uint8_image,tf.float32)
    return image,label

def preprocessing(image):
    # Randomly crop a [height, width] section of the image.
    height = IMAGAGE_SIZE
    width = IMAGAGE_SIZE
    distorted_image = tf.random_crop(image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    return float_image

def image_batcher(image, label, min_queue_examples,batch_size, shuffle):
    num_preprocess_threads = 16

    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])

def input_pipeline(data_dir,batch_size,eval_data = False):
    if not eval_data:
        filenames = [os.path.join(data_dir,"data_batch_%d.bin" % i ) for i in xrange(1,6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TEST

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    image,label = read_cifar10(filename_queue)

    image = preprocessing(image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return image_batcher(image,label,min_queue_examples,batch_size,shuffle=False)






