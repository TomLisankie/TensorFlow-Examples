import tensorflow as tf

# TensorFlow automatically figures out which variables others are dependent on and evaluates them as needed
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
    print(y.eval())
    print(z.eval())

# However, the previous method recomputes `w` and `x` with each evaluation
# If we want to do this more efficiently, we can compute `y` and `z` with each other.
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)