inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
		[0.5, -0.91, 0.26, -0.5],
		[-0.26, -0.27, 0.17, 0.87]]

biases = [ 2, 3, 0.5]

layer_outputs = [] # output of current layer

# neuron weights are paired with neuron biases 
# loops through all of the neurons in the output layer
# this computes a whole layer in python
for neuron_weights, neuron_bias in zip(weights, biases):
	neuron_output = 0 # output of given neuron
	# print('neuron weights: {}'.format(neuron_weights))
	# print('neuron bias: {}'.format(neuron_bias))

	# loop through all inputs and weights that are connecting that neuron
	# zips inputs and weights for that neuron 
	for n_input, weight in zip(inputs, neuron_weights):
		neuron_output += n_input*weight # sum all inputs * weights 


	neuron_output += neuron_bias # add the bias for that specfic neuron 
	layer_outputs.append(neuron_output) # append to the output

print(layer_outputs) # display output to screen