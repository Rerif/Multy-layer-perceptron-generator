import numpy as np

sigmoid = lambda x: (1/(1+2.718**(-x)))


class Multy_layer_perceptron():
	''' In the class passed list of count layers and neurons,
	and learning rate.
	example:
	plol = Perceptron_lot_of_layers([2, 3, 1], 0.1)'''
	def __init__(self, list_layers, learning_rate):
		if len(list_layers) < 3:
			assert False, "Using less than 3 layers"
		for neurons in list_layers:
			if neurons <= 0:
				assert False, "Can't use <= 0 count of neurons"
		''' creating input, hidden and output layers, learning rate and weights'''
		self.in_layer = list_layers[0]
		self.hidden_layers = list_layers[1: -1]
		self.output_lauer = list_layers[-1]
		self.lr = learning_rate
		self.weights = []
		# setting start random weights
		# weights between input and first hidden layer
		self.weights.append(np.random.rand(self.hidden_layers[0], self.in_layer)-0.5)
		i = 1
		# weights between hidden layers
		while i <= len(self.hidden_layers)-1:
			self.weights.append((np.random.rand(self.hidden_layers[i], self.hidden_layers[i-1]) - 0.5))
			i += 1
		# weights between last hidden and output layers
		self.weights.append((np.random.rand(self.output_lauer, self.hidden_layers[-1]) - 0.5))

	def perceptron_training(self, input_list, true_data_list):
		# data for training(recomendation (0...1))
		inputs = np.array(input_list, ndmin=2).T
		# true values for training
		targets = np.array(true_data_list, ndmin=2).T
		# hidden signals calculation
		hidden_signals = []
		i = 0
		while i < len(self.weights):
			if i == 0:
				hidden_signals.append(sigmoid(np.dot(self.weights[i], inputs)))
			else:
				hidden_signals.append(sigmoid(np.dot(self.weights[i], hidden_signals[i-1])))
			i += 1
		# errors calculations
		output_errors = targets - hidden_signals[-1]
		hidden_errors = []
		i = 0
		while i < len(self.weights) - 1:
			if i == 0:
				hidden_errors.append(np.dot(self.weights[-1].T, output_errors))
			else:
				hidden_errors.append(np.dot(self.weights[-1-i].T, hidden_errors[-1]))
			i += 1
		# distributed in proportion of weights
		# and recombined on hidden signals
		i = 0
		while i <= len(hidden_errors):
			if i == 0:
				self.weights[-1] += self.lr * np.dot((output_errors * hidden_signals[-1] * 
					(1.0 - hidden_signals[-1])), np.transpose(hidden_signals[-2]))
			elif i == len(hidden_errors):
				self.weights[-i-1] += self.lr * np.dot((hidden_errors[-i] * hidden_signals[-i-1] * 
					(1.0 - hidden_signals[-i-1])), np.transpose(inputs))
			else:
				self.weights[-i-1] += self.lr * np.dot((hidden_errors[-i] * hidden_signals[-i-1] * 
					(1.0 - hidden_signals[-i-1])), np.transpose(hidden_signals[-i - 2]))
			i += 1

	def use_perceptron(self, inputs_list):
		inputs = np.array(inputs_list, ndmin=2).T
		hidden_signals = []
		for i in range(len(self.weights)):
			if i == 0:
				hidden_signals.append(sigmoid(np.dot(self.weights[i], inputs)))
			else:
				hidden_signals.append(sigmoid(np.dot(self.weights[i], hidden_signals[-1])))
		return hidden_signals[-1][0][0]


def test():
	data_x = [0.1, 0.2, 0.3, 0.4, 0.5]
	data_y = [0.2, 0.3, 0.4, 0.5, 0.6]
	nn = Multy_layer_perceptron([1, 2, 3, 2, 1], 0.01)
	for epoch in range(10):
		for i in range(len(data_x)):
			true_x = data_x[i]
			true_y = data_y[i]
			nn.perceptron_training([true_x], [true_y])
	print(nn.use_perceptron([0.35]))


if __name__ == '__main__':
	test()
