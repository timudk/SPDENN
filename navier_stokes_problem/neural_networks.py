import tensorflow as tf 
import numpy as np 


class neural_networks:
	def __init__(self,
		 		 n_input_velocity,
				 n_input_pressure,
				 n_hidden_units_velocity,
				 n_hidden_units_pressure,
				 n_output_velocity,
				 n_output_pressure,
				 weight_initialization_velocity=tf.contrib.layers.xavier_initializer(),
				 weight_initialization_pressure=tf.contrib.layers.xavier_initializer(),
				 activation_hidden_velocity=tf.nn.sigmoid,
				 activation_hidden_pressure=tf.nn.sigmoid):

		self.nn_velocity = velocity(n_input_velocity,
									n_hidden_units_velocity,
									n_output_velocity, 
									weight_initialization_velocity,
									activation_hidden_velocity)

		self.nn_pressure = pressure(n_input_pressure,
									n_hidden_units_pressure,
			                        n_output_pressure, 
			                        weight_initialization_pressure,
			                        activation_hidden_pressure)


class velocity:
	def __init__(self,
				 n_input,
				 n_hidden_layers_and_units,
				 n_output, 
				 weight_initialization,
				 activation_hidden):

		self.n_input = n_input
		self.n_hidden_layers_and_units = n_hidden_layers_and_units
		self.n_output = n_output
		self.weight_initialization = weight_initialization
		self.activation_hidden = activation_hidden

		self.weights = {}
		self.biases = {}

		self.number_of_layers = len(self.n_hidden_layers_and_units)

		for i in range(0, self.number_of_layers):
			if i == 0:
				self.weights['0'] = tf.get_variable('velocity_weight_' + str(0), shape=[self.n_input, self.n_hidden_layers_and_units[0]], initializer=self.weight_initialization, dtype=tf.float64)
			else:
				self.weights[str(i)] = tf.get_variable('velocity_weight_' + str(i), shape=[self.n_hidden_layers_and_units[i-1], self.n_hidden_layers_and_units[i]], initializer=self.weight_initialization, dtype=tf.float64)

			self.biases[str(i)] = tf.get_variable('velocity_bias_' + str(i), shape=self.n_hidden_layers_and_units[i], initializer=self.weight_initialization, dtype=tf.float64)

		self.weights[str(self.number_of_layers)] =  tf.get_variable('velocity_weight_' + str(self.number_of_layers), shape=[self.n_hidden_layers_and_units[-1], self.n_output], initializer=self.weight_initialization, dtype=tf.float64)
		self.biases[str(self.number_of_layers)] =tf.get_variable('velocity_bias_' + str(self.number_of_layers), shape=[self.n_output], initializer=self.weight_initialization, dtype=tf.float64)


	def value_nn_velocity(self, velocity_var):
		for i in range(0, self.number_of_layers):
			if i == 0:	
				layer = tf.add(tf.matmul(velocity_var, self.weights['0']), self.biases['0'])
			else: 
				layer = tf.add(tf.matmul(layer, self.weights[str(i)]), self.biases[str(i)])

			# print(layer)
			layer = self.activation_hidden(layer)

		return tf.matmul(layer, self.weights[str(self.number_of_layers)]) + self.biases[str(self.number_of_layers)]

	def first_derivatives_nn_velocity(self, X):
		batch_size = tf.shape(X)[0]

		if(self.n_input==1):
			return self.first_derivatives_nn_velocity_1d(X, batch_size)
		else:
			#return self.first_derivatives_nn_velocity_2d(X, batch_size)
			return self.first_derivates_nn_velocity_multidimensional(X, batch_size)

	def first_derivates_nn_velocity_multidimensional(self, X, batch_size):
		grad_velocity = []

		for i in range(self.n_input):
			grad_velocity.append(tf.gradients(tf.slice(self.value_nn_velocity(X), [0,i], [batch_size,1]), X))

		# return grad_velocity[0][0], grad_velocity[1][0]
		return grad_velocity

	def first_derivatives_nn_velocity_1d(self, X, batch_size):
		grad_u = tf.gradients(self.value_nn_velocity(X), X)
		return grad_u[0]

	# def first_derivatives_nn_velocity_2d(self, X, batch_size):
	# 	grad_u = tf.gradients(tf.slice(self.value_nn_velocity(X), [0,0], [batch_size,1]), X)
	# 	grad_v = tf.gradients(tf.slice(self.value_nn_velocity(X), [0,1], [batch_size,1]), X)
	# 	return grad_u[0], grad_v[0]

	def second_derivatives_nn_velocity(self, X):
		batch_size = tf.shape(X)[0]

		if(self.n_input==1):
			return self.second_derivatives_nn_velocity_1d(X, batch_size)
		else:
			# return self.second_derivatives_nn_velocity_2d(X, batch_size)
			return self.second_derivatives_nn_velocity_multidimensional(X, batch_size)

	def second_derivatives_nn_velocity_multidimensional(self, X, batch_size):
		grad_velocity = self.first_derivates_nn_velocity_multidimensional(X, batch_size)
		# print(len(grad_velocity))

		grad_grad_velocity = []
		for i in range(len(grad_velocity)):
			second_derivatives = []
			for j in range(self.n_input):
				second_derivatives.append(tf.gradients(tf.slice(grad_velocity[i][0], [0, j], [batch_size,1]), X))

			grad_grad_velocity.append(second_derivatives)

		# print(len(grad_grad_velocity[0]))

		return grad_grad_velocity

	def second_derivatives_nn_velocity_1d(self, X, batch_size):
		grad_u = self.first_derivatives_nn_velocity_1d(X, batch_size)

		grad_u_x = tf.gradients(tf.slice(grad_u, [0,0], [batch_size,1]), X)
		return grad_u_x[0]

	def second_derivatives_nn_velocity_2d(self, X, batch_size):
		grad_u, grad_v = self.first_derivatives_nn_velocity_2d(X, batch_size)

		grad_u_x = tf.gradients(tf.slice(grad_u, [0,0], [batch_size,1]), X)
		grad_u_y = tf.gradients(tf.slice(grad_u, [0,1], [batch_size,1]), X)

		grad_v_x = tf.gradients(tf.slice(grad_v, [0,0], [batch_size,1]), X)
		grad_v_y = tf.gradients(tf.slice(grad_v, [0,1], [batch_size,1]), X)
		
		return grad_u_x[0], grad_u_y[0], grad_v_x[0], grad_v_y[0]

class pressure:
	def __init__(self,
				 n_input,
				 n_hidden_layers_and_units,
				 n_output, 
				 weight_initialization,
				 activation_hidden):

		self.n_input = n_input
		self.n_hidden_layers_and_units = n_hidden_layers_and_units
		self.n_output = n_output
		self.weight_initialization = weight_initialization
		self.activation_hidden = activation_hidden

		self.weights = {}
		self.biases = {}

		self.number_of_layers = len(self.n_hidden_layers_and_units)
		# print(self.number_of_layers)

		for i in range(0, self.number_of_layers):
			if i == 0:
				self.weights['0'] = tf.get_variable('pressure_weight_' + str(0), shape=[self.n_input, self.n_hidden_layers_and_units[0]], initializer=self.weight_initialization, dtype=tf.float64)
			else:
				self.weights[str(i)] = tf.get_variable('pressure_weight_' + str(i), shape=[self.n_hidden_layers_and_units[i-1], self.n_hidden_layers_and_units[i]], initializer=self.weight_initialization, dtype=tf.float64)

			self.biases[str(i)] = tf.get_variable('pressure_bias_' + str(i), shape=self.n_hidden_layers_and_units[i], initializer=self.weight_initialization, dtype=tf.float64)


		self.weights[str(self.number_of_layers)] =  tf.get_variable('pressure_weight_' + str(self.number_of_layers), shape=[self.n_hidden_layers_and_units[-1], self.n_output], initializer=self.weight_initialization, dtype=tf.float64)
		self.biases[str(self.number_of_layers)] =tf.get_variable('pressure_bias_' + str(self.number_of_layers), shape=[self.n_output], initializer=self.weight_initialization, dtype=tf.float64)

	def value_nn_pressure(self, pressure_var):
		for i in range(0, self.number_of_layers):
			if i == 0:	
				layer = tf.add(tf.matmul(pressure_var, self.weights['0']), self.biases['0'])
			else: 
				layer = tf.add(tf.matmul(layer, self.weights[str(i)]), self.biases[str(i)])

			layer = self.activation_hidden(layer)

		return tf.matmul(layer, self.weights[str(self.number_of_layers)]) + self.biases[str(self.number_of_layers)]

	# def first_derivatives_nn_pressure(self, X):
	# 	batch_size = tf.shape(X)[0]
	# 	grad_p = tf.gradients(self.value_nn_pressure(X), X)

	# 	p_x = tf.slice(grad_p[0], [0,0], [batch_size,1])
	# 	p_y = tf.slice(grad_p[0], [0,1], [batch_size,1])

	# 	return p_x, p_y

	def first_derivates_nn_pressure_multidimensional(self, X):
		batch_size = tf.shape(X)[0]
		grad_pressure = []

		for i in range(self.n_input):
			grad_pressure.append(tf.gradients(tf.slice(self.value_nn_pressure(X), [0,i], [batch_size,1]), X))

		return grad_pressure

	# def grad_nn_pressure(self, X):
	# 	return tf.gradients(self.value_nn_pressure(X), X)