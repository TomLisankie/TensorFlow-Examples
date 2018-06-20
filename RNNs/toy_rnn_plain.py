'''
A toy RNN just using numpy
'''

import numpy as np

timesteps = 100 # number of elements per sequence
input_features = 32 # number of features for each sequence element vector
output_features = 64 # number of features for each output vector

inputs = np.random.random((timesteps, input_features))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features))

state_t = np.zeros((output_features))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

final_output_sequence = np.concatenate(successive_outputs, axis = 0)
print(final_output_sequence)