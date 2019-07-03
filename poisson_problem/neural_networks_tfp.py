import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np

class neural_network:
	def __init__(self,
				 n_input,
				 n_output,
		 		 n_hidden_units,
				 weight_initialization=tf.contrib.layers.xavier_initializer(), 
				 activation_hidden=tf.nn.sigmoid,
				 name='velocity_',
				 starting_vector=None,
				 tol=1e-8,
				 dtype=tf.float32,
				 max_itr=100):

		self.int_var = tf.placeholder(dtype, [None, n_input]) 
		self.bou_var = tf.placeholder(dtype, [None, n_input]) 

		self.sol_int = tf.placeholder(dtype, [None, 1])
		self.sol_bou = tf.placeholder(dtype, [None, 1])

		self.n_input = n_input
		self.n_output = n_output
		self.n_hidden_units = n_hidden_units
		self.weight_initialization = weight_initialization
		self.activation_hidden = activation_hidden

		self.weights = {}
		self.biases = {}
		self.number_of_layers = len(self.n_hidden_units)

		self.name = name

		for i in range(0, self.number_of_layers):
			if i == 0:
				self.weights[self.name+'0'] = tf.get_variable(self.name + 'weight_' + str(0), shape=[self.n_input, self.n_hidden_units[0]], initializer=self.weight_initialization, dtype=dtype)
			else:
				self.weights[self.name+str(i)] = tf.get_variable(self.name + 'weight_' + str(i), shape=[self.n_hidden_units[i-1], self.n_hidden_units[i]], initializer=self.weight_initialization, dtype=dtype)

			self.biases[self.name+str(i)] = tf.get_variable(self.name + 'bias_' + str(i), shape=[1, self.n_hidden_units[i]], initializer=self.weight_initialization, dtype=dtype)

		self.weights[self.name+str(self.number_of_layers)] =  tf.get_variable(self.name+'weight_' + str(self.number_of_layers), shape=[self.n_hidden_units[-1], self.n_output], initializer=self.weight_initialization, dtype=dtype)
		self.biases[self.name+str(self.number_of_layers)] =tf.get_variable(self.name+'bias_' + str(self.number_of_layers), shape=[1, self.n_output], initializer=self.weight_initialization, dtype=dtype)

		if starting_vector is None:
			print('Number of trainable parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
			self.starting_vector = tf.get_variable('starting_vector', shape=[np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]), 1], dtype=dtype, initializer=tf.initializers.random_normal)
		self.tol = tol
		self.max_itr = max_itr

	def value(self, input_var):
		for i in range(0, self.number_of_layers):
			if i == 0:	
				layer = tf.add(tf.matmul(input_var, self.weights[self.name+'0']), self.biases[self.name+'0'])
			else: 
				layer = tf.add(tf.matmul(layer, self.weights[self.name+str(i)]), self.biases[self.name+str(i)])

			layer = self.activation_hidden(layer)

		return tf.add(tf.matmul(layer, self.weights[self.name+str(self.number_of_layers)]), self.biases[self.name+str(self.number_of_layers)])


	def first_derivatives(self, X):
		return tf.gradients(self.value(X), X)[0]

	def second_derivatives(self, X):
		grad = self.first_derivatives(X)
		grad_grad = []

		for i in range(self.n_input):
			grad_grad.append(tf.slice(tf.gradients(tf.slice(grad, [0, i], [tf.shape(X)[0], 1]), X)[0], [0, i],  [tf.shape(X)[0], 1]))

		return grad_grad

	def bfgs_tfp(self):

		return tfp.optimizer.bfgs_minimize(self.function_evaluation, initial_position=self.starting_vector, tolerance=self.tol, max_iterations=self.max_itr)

	def function_evaluation(self, coord):
		self.get_split(coord)

		gradient_list = self.compute_gradients()
		gradients = tf.concat(gradient_list, axis=0)

		return self.loss(), gradients

	def get_split(self, weights):
		weight_list = self.compute_split()

		split = tf.split(weights, weight_list, 0)

		counter = 0
		for i in self.weights:
			self.weights[i] = tf.reshape(split[counter], tf.shape(self.weights[i]))
			counter += 1

		for i in self.biases:
			self.biases[i] = tf.reshape(split[counter], tf.shape(self.biases[i]))
			counter += 1

	def compute_split(self):
		#layer split
		l = []
		for i in self.weights:
			l.append(tf.shape(self.weights[i])[0]*tf.shape(self.weights[i])[1])

		#bias split
		b = []
		for i in self.biases:
			b.append(tf.shape(self.biases[i])[0]*tf.shape(self.biases[i])[1])

		return l + b

	def compute_gradients(self):
		weight_list = self.compute_split()
		counter = 0

		l_gradients = []
		for i in self.weights:
			l_gradients.append(tf.reshape(tf.gradients(self.loss(), self.weights[i])[0], [weight_list[counter], 1]))
			counter += 1

		#bias split
		b_gradients = []
		for i in self.biases:
			b_gradients.append(tf.reshape(tf.gradients(self.loss(), self.biases[i])[0], [weight_list[counter], 1]))
			counter += 1

		return l_gradients + b_gradients

	def loss(self):
		value_bou = self.value(self.bou_var)

		grad = self.first_derivatives(self.int_var)
		grad_grad= self.second_derivatives(self.int_var)
		sum_of_second_derivatives = 0.0
		for i in range(self.n_input):
			sum_of_second_derivatives += grad_grad[i]

		loss_int = tf.square(sum_of_second_derivatives + self.sol_int)
		loss_bou = tf.square(value_bou - self.sol_bou)

		return tf.sqrt(tf.reduce_mean(loss_int + loss_bou))
		# return tf.sqrt(tf.reduce_mean(loss_bou))
