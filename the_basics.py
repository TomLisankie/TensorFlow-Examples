import tensorflow as tf 

# Create computation graph
x = tf.Variable(3, name = "x")
y = tf.Variable(4, name = "y")
f = x*x*y + y + 2

# Create session, initialize variables, and run session on variable `f`
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

# Better way to run a session. Also, session is automatically closed.
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

# Execute a session without having to initialize each variable individually
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
    print(result)