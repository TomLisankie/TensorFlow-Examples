import tensorflow as tf 

# Any node created is automatically added to the default graph
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

# If you want multiple, independent graphs you can create them.
# Assign variables to a graph by temporarily making the graph the default.
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())