import tensorflow as tf 
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Manually compute the gradients
n_epochs = 1000
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(
    housing.data)
housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

X = tf.constant(housing_data_plus_bias, dtype = tf.float32, name = "X")
y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name = "y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name = "theta")
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = "mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    print("Best Theta:", best_theta)

# We don't have to implement gradient descent by hand though, we can just use TF's autodiff
gradients = tf.gradients(mse, [theta])[0]

# It gets even easier though. TF has a bunch of optimizers out of the box.
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)