import numpy as np

# dot product for a layer of neurons

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
		[0.5, -0.91, 0.26, -0.5],
		[-0.26, -0.27, 0.17, 0.87]]

biases = [ 2, 3, 0.5]


# first element you pass is how the element will be indexed, we want it to be indexed by the weights
# since weights is a matrix of three vectors, we are doing the dot product of each vector (results in three dot products)
output = np.dot(weights, inputs) + biases

# with inputs first we get a shape error
# output = np.dot(inputs, weights) + biases

# the reason why is because we are iterating through a matrix of vectors


print(output)