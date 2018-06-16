import tensorflow as tf 
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Manually compute the gradients
n_epochs = 1000
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(
    housing.data)
housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# Placeholder nodes don't perform any computation. They just exist to output data at runtime
X = tf.placeholder(dtype = tf.float32, shape = (None, n + 1), name = "X")
y = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = "y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name = "theta")
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = "mse")
gradients = tf.gradients(mse, [theta])[0]
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)

def fetch_batch(epoch, batch_index, batch_size):
    # load data from disk
    # ...
    return X_batch, y_batch

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
    best_theta = theta.eval()
    print("Best Theta:", best_theta)

    # Save the model
    save_path = saver.save(sess, "/tmp/my_model.ckpt")

