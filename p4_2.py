import numpy as np

np.random.seed(0)


# use X to denote input feature sets 
# this is our training data set, input to nueral network
X = [[1, 2, 3, 2.5],
	[2.0, 5.0, -1.0, 2.0],
	[-1.5, 2.7, 3.3, -0.8]]

# specify hidden layers 
# we are not in charge of how the layer changes 


# to initalize layer, there are two ways
# 1 use a trained model that you saved, we are setting the weights and biases for that model
# 2 we are using a new nueral network, we have weights and biases that need to be initialized 
# we want small values so things stay in range of +1 to -1

# In the case with our input data set we need to normalize and scale the values so it can be used for the dataset
# our weights should be between -0.1 and 0.1


# 0 biases = dead network
# solution start biases at non zero number

class Layer_Dense():
	def __init__(self, n_inputs, n_neurons):
		# need to know what the size of the input is coming in 
		# and how many nueorns we want to have

		# randn - gaussian distribution bounded around 0
		self.weights =  0.1 * np.random.randn(n_inputs, n_neurons) # makes the shapes for our weights matrix
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

# print('layer1 output')
layer1.forward(X)
# print('layer1 weights: {}'.format(layer1.weights))
# print('layer1 biases: {}'.format(layer1.biases))
# print(layer1.output)

print('layer2 output')
layer2.forward(layer1.output)
print(layer2.output)