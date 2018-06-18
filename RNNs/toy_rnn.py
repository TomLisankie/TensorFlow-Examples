'''
Simple RNN written from scratch that runs over only 2 timesteps.
'''

import tensorflow as tf 

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs]) # input 1
X1 = tf.placeholder(tf.float32, [None, n_inputs]) # input 2

Wx = tf.Variable(tf.random_normal(shape = [n_inputs, n_neurons], dtype = tf.float32)) # input weights
Wy = tf.Variable(tf.random_normal(shape = [n_neurons, n_neurons], dtype = tf.float32)) # recurrent weights
b = tf.Variable(tf.zeros([1, n_neurons], dtype = tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b) # first output
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b) # first recurrent input

init = tf.global_variables_initializer()

import numpy as np 

# Mini-batches      #instance 0, instance 1, etc.
X0_batch = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]
X1_batch = [[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict = {X0: X0_batch, X1: X1_batch})

print(Y0_val)
print(Y1_val)