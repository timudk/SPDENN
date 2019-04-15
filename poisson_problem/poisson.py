import tensorflow as tf 
import numpy as np 
from scipy import integrate
import neural_networks
import poisson_problem
import matplotlib.pyplot as plt
import sys, getopt

class sampling_from_dataset:
	def __init__(self, filepath, total_samples):
		self.filepath = filepath
		self.total_samples = total_samples

		self.last_grab_int = 0
		self.last_grab_bou = 0

	def load_dataset(self):
		self.dataset = np.genfromtxt(self.filepath, delimiter=',')

	def increase_grab_number(self, num, batchsize):
		num += batchsize
		if(num==self.total_samples):
			return 0
		else:
			return num

	def interior_samples(self, batchsize):
		sampling_int_draw_x = self.dataset[self.last_grab_int:(self.last_grab_int+batchsize), 0]
		sampling_int_draw_y = self.dataset[self.last_grab_int:(self.last_grab_int+batchsize), 1]

		self.last_grab_int = self.increase_grab_number(self.last_grab_int, batchsize)

		return sampling_int_draw_x, sampling_int_draw_y
	def boundary_samples(self, batchsize):
		sampling_bou_draw_x = self.dataset[self.last_grab_bou:(self.last_grab_bou+batchsize), 2]
		sampling_bou_draw_y = self.dataset[self.last_grab_bou:(self.last_grab_bou+batchsize), 3]

		self.last_grab_bou = self.increase_grab_number(self.last_grab_bou, batchsize)

		return sampling_bou_draw_x, sampling_bou_draw_y


def main(argv):

	# DEFAULT
	SENSOR_DATA = False
	N_LAYERS = 1
	BATCHSIZE = 1000
	max_iter = 20000
	seed = 42

	try:
		opts, args = getopt.getopt(argv,"hb:n:m:s:r:",["batchsize=","n_layers=", "max_iterations=", "sensor_data=", "random_seed="])
	except getopt.GetoptError:
		print('poisson.py -b <batchsize> -n <n_layers> -m <max_iterations> -s <sensor_data> -r <random_seed>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
	 		print('poisson.py -b <batchsize> -n <n_layers> -m <max_iterations> -s <sensor_data> -r <random_seed>')
	 		sys.exit()
		elif opt in ("-b", "--batchsize"):
	 		BATCHSIZE = int(arg)
		elif opt in ("-n", "--n_layers"):
	 		N_LAYERS = int(arg)
		elif opt in ("-m", "--max_iterations"):
			max_iter = int(arg)
		elif opt in ("-s", "--sensor_data"):
			if(int(arg)==1):
				SENSOR_DATA = True
		elif opt in ("-r", "--random_seed"):
			seed = int(arg)
			tf.set_random_seed(seed)

	HIDDEN_UNITS = []
	for i in range(N_LAYERS):
		HIDDEN_UNITS.append(16)

	do_save = True
	if(SENSOR_DATA):
		save_name = 'test_model/' + str(len(HIDDEN_UNITS)) + '_layer_sq_loss_' + str(BATCHSIZE) + '_m_iter_' + str(max_iter) + '_rs_' + str(seed) + '_wsd'
	else:
		save_name = 'test_model/' + str(len(HIDDEN_UNITS)) + '_layer_sq_loss_' + str(BATCHSIZE) + '_m_iter_' + str(max_iter) + '_rs_' + str(seed) 

	problem = poisson_problem.poisson_2d()
	
	sampler = sampling_from_dataset('datasets/' + str(BATCHSIZE), BATCHSIZE)
	sampler.load_dataset()

	NUM_INPUTS = 2
	neural_network = neural_networks.neural_network(NUM_INPUTS, 1, HIDDEN_UNITS)


	int_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 
	bou_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 
	sensor_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 
	

	value_int = neural_network.value(int_var)
	value_bou = neural_network.value(bou_var)
	value_sensor = neural_network.value(sensor_var)

	grad = neural_network.first_derivatives(int_var)
	grad_grad= neural_network.second_derivatives(int_var)

	grad_grad_sensor = neural_network.second_derivatives(sensor_var)

	sol_int = tf.placeholder(tf.float64, [None, 1])
	sol_bou = tf.placeholder(tf.float64, [None, 1])

	sum_of_second_derivatives = 0.0
	sum_of_second_derivatives_sensor = 0.0
	for i in range(NUM_INPUTS):
		sum_of_second_derivatives += grad_grad[i]
		sum_of_second_derivatives_sensor += grad_grad_sensor[i]


	loss_int = tf.square(sum_of_second_derivatives+sol_int)
	loss_bou = tf.square(value_bou-sol_bou)

	loss_sensor_int = tf.square(sum_of_second_derivatives_sensor)
	loss_sensor_bou = tf.square(value_sensor)

	loss = tf.sqrt(tf.reduce_mean(loss_int + loss_bou))
	sensor_loss = tf.sqrt(tf.reduce_mean(loss_int) + tf.reduce_mean(loss_bou) + tf.reduce_mean(loss_sensor_int) + tf.reduce_mean(loss_sensor_bou))

	train_scipy = tf.contrib.opt.ScipyOptimizerInterface(loss, method='BFGS', options={'gtol':1e-14, 'disp':True, 'maxiter':max_iter})
	train_scipy_sensor = tf.contrib.opt.ScipyOptimizerInterface(sensor_loss, method='BFGS', options={'gtol':1e-14, 'disp':True, 'maxiter':max_iter})

	
	init = tf.global_variables_initializer()

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)
		
		int_draw_x, int_draw_y = sampler.interior_samples(BATCHSIZE)
		int_draw_x = np.reshape(int_draw_x, (BATCHSIZE, 1)) 
		int_draw_y = np.reshape(int_draw_y, (BATCHSIZE, 1))

		boundary_draw_x, boundary_draw_y = sampler.boundary_samples(BATCHSIZE)
		boundary_draw_x = np.reshape(boundary_draw_x, (BATCHSIZE, 1))
		boundary_draw_y = np.reshape(boundary_draw_y, (BATCHSIZE, 1))


		int_draw = np.concatenate([int_draw_x, int_draw_y], axis=1)
		bou_draw = np.concatenate([boundary_draw_x, boundary_draw_y], axis=1)


		f = problem.rhs(int_draw)
		f = np.reshape(np.array(f), (BATCHSIZE, 1))

		bou = problem.velocity(bou_draw)
		bou = np.reshape(np.array(bou), (BATCHSIZE, 1))

		if(SENSOR_DATA):
			sensor_points_x = np.reshape(np.array([0.0, 1.0, 0.0, 1.0]), (4,1))
			sensor_points_y = np.reshape(np.array([0.0, 0.0, 1.0, 1.0]), (4,1))

			sensor_points = np.concatenate([sensor_points_x, sensor_points_y], axis=1)
			print(sensor_points)

			train_scipy_sensor.minimize(sess, feed_dict={sol_int:f, sol_bou:bou, int_var:int_draw, bou_var:bou_draw, sensor_var: sensor_points})
		else:
			train_scipy.minimize(sess, feed_dict={sol_int:f, sol_bou:bou, int_var:int_draw, bou_var:bou_draw})
	

		if do_save:		
			save_path = saver.save(sess, save_name)
			print("Model saved in path: %s" % save_path)


if __name__ == '__main__':
	main(sys.argv[1:])
