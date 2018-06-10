import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  ### Building a model with 2 Convolution layers
  ### followed by 2 fully connected hidden layers and a linear classification layer.
  x = tf.placeholder(tf.float32, [None, 784])

  # Parameters for the 2 convolution layer
  with tf.variable_scope("CNN"):
      cnn_W1 = tf.get_variable("W1", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
      cnn_b1 = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.1))
      cnn_W2 = tf.get_variable("W2", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
      cnn_b2 = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0.1))

  # Parameters for 2 hidden layers with dropout and the linear classification layer.
  # 3136 = 7 * 7 * 64
  W1 = tf.get_variable("W1", [3136, 1000], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 3136)))
  b1 = tf.get_variable("b1", [1000], initializer=tf.zeros_initializer)
  W2 = tf.get_variable("W2", [1000, 100], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 1000)))
  b2 = tf.get_variable("b2", [100], initializer=tf.zeros_initializer)
  W3 = tf.get_variable("W3", [100, 10], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 100)))
  b3 = tf.get_variable("b3", [10], initializer=tf.zeros_initializer)

  keep_prob = tf.placeholder(tf.float32)

  # First CNN with RELU and max pooling.
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  cnn1 = tf.nn.conv2d(x_image, cnn_W1, strides=[1, 1, 1, 1], padding='SAME')
  z1 = tf.nn.relu(cnn1 + cnn_b1)
  h1 = tf.nn.max_pool(z1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Second CNN
  cnn2 = tf.nn.conv2d(h1, cnn_W2, strides=[1, 1, 1, 1], padding='SAME')
  z2 = tf.nn.relu(cnn2 + cnn_b2)
  h2 = tf.nn.max_pool(z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # First FC layer with dropout.
  h2_flat = tf.reshape(h2, [-1, 3136])
  h_fc1 = tf.nn.relu(tf.matmul(h2_flat, W1) + b1)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Second FC
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W2) + b2)
  h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  # Linear classification.
  y = tf.matmul(h_fc2_drop, W3) + b3

  # True label
  labels = tf.placeholder(tf.float32, [None, 10])

  # Cost function & optimizer
  # Use cross entropy with the Adam gradient descent optimizer.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y) )
  train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(init)
    # Train
    for i in range(10001):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, keep_prob:0.5})
      print(f"Iteration {i}")	
      saver.save(sess,'/tmp/tensorflowcnn.ckpt')
	
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
