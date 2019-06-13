import tensorflow as tf 
tf.set_random_seed(42)

import numpy as np 
from scipy import integrate
import neural_networks_tfp
import poisson_problem
import matplotlib.pyplot as plt
import sys, getopt
import tensorflow_probability as tfp

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
	MAX_ITER = 50000
	DO_SAVE = False
	SEED = 42

	try:
		opts, args = getopt.getopt(argv,"hb:n:m:d:r:s:",["batchsize=","n_layers=", "max_iterations=", "sensor_data=", "random_seed=", "save_network="])
	except getopt.GetoptError:
		print('poisson.py -b <batchsize> -n <n_layers> -m <max_iterations> -d <sensor_data> -r <random_seed> -s <save_network>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
	 		print('poisson.py -b <batchsize> -n <n_layers> -m <max_iterations> -d <sensor_data> -r <random_seed> -s <save_network>')
	 		sys.exit()
		elif opt in ("-b", "--batchsize"):
	 		BATCHSIZE = int(arg)
		elif opt in ("-n", "--n_layers"):
	 		N_LAYERS = int(arg)
		elif opt in ("-m", "--max_iterations"):
			MAX_ITER = int(arg)
		elif opt in ("-d", "--sensor_data"):
			if(int(arg)==1):
				SENSOR_DATA = True
		elif opt in ("-r", "--random_seed"):
			SEED = int(arg)
			tf.set_random_seed(SEED)
		elif opt in ("-s", "--save_network"):
			DO_SAVE = bool(int(arg))
			if DO_SAVE:
				print("Saving network after training.")

	HIDDEN_UNITS = []
	for i in range(N_LAYERS):
		HIDDEN_UNITS.append(16)

	if(SENSOR_DATA):
		save_name = 'test_model/' + str(len(HIDDEN_UNITS)) + '_layer_sq_loss_' + str(BATCHSIZE) + '_m_iter_' + str(MAX_ITER) + '_rs_' + str(SEED) + '_wsd'
	else:
		save_name = 'test_model/' + str(len(HIDDEN_UNITS)) + '_layer_sq_loss_' + str(BATCHSIZE) + '_m_iter_' + str(MAX_ITER) + '_rs_' + str(SEED) 

	problem = poisson_problem.poisson_2d()
	
	sampler = sampling_from_dataset('datasets/' + str(BATCHSIZE), BATCHSIZE)
	sampler.load_dataset()

	NUM_INPUTS = 2
	neural_network = neural_networks_tfp.neural_network(NUM_INPUTS, 1, HIDDEN_UNITS)

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

		result = sess.run(neural_network.bfgs_tfp(), feed_dict={neural_network.sol_int:f, neural_network.sol_bou:bou, neural_network.int_var:int_draw, neural_network.bou_var:bou_draw})
		print ('Function evaluations: %d' % result.num_objective_evaluations)
		print('Failed?', result.failed)
		print('Converged?', result.converged)

		if DO_SAVE:		
			save_path = saver.save(sess, save_name)
			print("Model saved in path: %s" % save_path)


if __name__ == '__main__':
	main(sys.argv[1:])
