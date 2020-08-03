import numpy as np

# dot product for a layer of neurons

# instead of a vector of inputs, we are using a batch of inputs so we can do multiple layers of nuerons
# soon we will make our own layer object

# we use batches because the calculations are simple(multiplying matricies)
# this way we can use GPUs since there are more cores and better for simpler calculations

# think of these inputs as features of a dataset, ex: they describe the status of a server 
# at a single point in time

# we need to pass the batch of these samples instead, so the model can generalize to multiple samples


# weights and biases are associated with neurons, if we do not change neurons then we do not change these values
inputs = [[1, 2, 3, 2.5],
		  [2.0, 5.0, -1.0, 2.0],
		  [-1.5, 2.7, 3.3, -0.8]]


# first layer
weights = [[0.2, 0.8, -0.5, 1.0],
		   [0.5, -0.91, 0.26, -0.5],
		   [-0.26, -0.27, 0.17, 0.87]]

biases = [ 2, 3, 0.5]



# secondlayer 
weights2  = [[0.1, -0.14, 0.5],
		   [-0.5, 0.12, -0.33],
		   [-0.44, 0.73, -0.13]]

biases2 = [ -1, 2, -0.5]

# we transpose the weights matrix so the shapes fit with the inputs
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)