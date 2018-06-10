import argparse
import sys
from PIL import Image
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):
    sess=tf.Session()
            
    with tf.Session() as sess:
        
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('/tmp/tensorflowcnn.ckpt.meta')
        
        saver.restore(sess,'/tmp/tensorflowcnn.ckpt')
        graph=tf.get_default_graph()
        
    
    

        cnn_W1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CNN')[0]
        cnn_b1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CNN')[1]
        cnn_W2=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CNN')[2]
        cnn_b2=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CNN')[3]
        
        
        cnn_W1 = tf.convert_to_tensor(cnn_W1, dtype=tf.float32)
        cnn_b1=tf.convert_to_tensor(cnn_b1, dtype=tf.float32)
        cnn_W2=tf.convert_to_tensor(cnn_W2, dtype=tf.float32)
        cnn_b2=tf.convert_to_tensor(cnn_b2, dtype=tf.float32)
        
        W1 = graph.get_tensor_by_name("W1:0")
        b1 = graph.get_tensor_by_name("b1:0")
        W2 = graph.get_tensor_by_name("W2:0")
        b2 = graph.get_tensor_by_name("b2:0")
        W3 = graph.get_tensor_by_name("W3:0")
        b3 = graph.get_tensor_by_name("b3:0")
        keep_prob = tf.placeholder(tf.float32)
        
        images=np.load('mnist_rotated_data.npy')
        labels=np.load('mnist_rotated_labels.npy')
        
        
        count=0
        for num in range(len(images)):
            
            img = images[num]
        
            img = img.reshape((28, 28))
         
            
        
         
            
            
            a=tf.cast(img,tf.float32)
        
        # First CNN with RELU and max pooling.
            x_image = tf.reshape(a, [ -1,28, 28,1])
         
          
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
            h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)
        
        
        # Second FC
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W2) + b2)
            h_fc2_drop = tf.nn.dropout(h_fc2, 1.0)
        
        # Linear classification.
            y = tf.matmul(h_fc2_drop, W3) + b3
         
            #print(f"Predicted: ",type(tf.argmax(y,1).eval()[0]))

##            print(type(tf.argmax(y,1)[0].eval()))
##            print(type(labels[num][0]))
            if tf.argmax(y,1)[0].eval() == int(labels[num][0]):
                count+=1

        print("accuracy=",count/10000)
                

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
