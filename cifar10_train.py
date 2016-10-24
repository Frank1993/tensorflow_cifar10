from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import os.path
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

import cifar10


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_dir","cifar10_data","""Directory where to write event logs and checkpoint.""")

tf.app.flags.DEFINE_integer("max_steps", 1000,"""Number of batches to run""")

def train():
	with tf.Graph().as_default():
		global_step = tf.Variable(0,trainable = False)

		#get images and labels for cifar-10
		images,labels = cifar10.inputs()

		logits = cifar10.inference(images)

		loss = cifar10.loss(logits,labels)

		train_op = cifar10.train(loss,global_step)

		with tf.Session() as sess:
			init = tf.initialize_all_variables()

			sess.run(init)

			tf.train.start_queue_runners()

			for step in xrange(FLAGS.max_steps):
				_,loss_value = sess.run([train_op,loss])

				if step % 10 == 0:
					print("Step %s: loss = %s"%(step,loss_value))

def main(argv = None):
	cifar10.maybe_download_and_extract()

	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)

	print(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)

	train()

if __name__ == "__main__":
	tf.app.run()
