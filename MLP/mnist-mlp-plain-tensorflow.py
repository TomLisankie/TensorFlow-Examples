import tensorflow as tf 
import numpy as np

'''
Construction Phase
'''
n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X") # acts as the input layer
y = tf.placeholder(tf.int64, shape = (None), name = "y")

# Creates a neuron layer
def neuron_layer(X, n_neurons, layer_name, activation = None):
    with tf.name_scope(layer_name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
        W = tf.Variable(init, name = "kernel") #Weights matrix is the same as the kernel
        b = tf.Variable(tf.zeros([n_neurons]), name = "bias")
        Z = tf.matmul(X, W) + b 
        if activation is not None:
            return activation(Z)
        else:
            return Z 
        
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name = "hidden1", activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name = "hidden2", activation = tf.nn.relu)
    # logits are the output of the network before going through softmax
    logits = neuron_layer(hidden2, n_outputs, name = "outputs") 

with tf.name_scope("loss"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(cross_entropy, name = "loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

'''
Execution Phase
'''
init = tf.global_variables_initializer()
saver = tf.train.Saver()