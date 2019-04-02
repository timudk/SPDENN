import tensorflow as tf 
tf.set_random_seed(42)

import numpy as np 
from scipy import integrate
import neural_networks
import kovasznay_flow_problem
import matplotlib.pyplot as plt
import quadpy
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
			print('New epoch!')
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

	try:
		opts, args = getopt.getopt(argv,"hb:u:p:m:r:",["batchsize=","n_layers_velocity=", "n_layers_pressure=","max_iterations=", "random_seed="])
	except getopt.GetoptError:
		print('kovasznay_flow.py -b <batchsize> -u <n_layers_velocity> -p <n_layers_pressure> -m <max_iterations>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
	 		print('kovasznay_flow.py -b <batchsize> -u <n_layers_velocity> -p <n_layers_pressure> -m <max_iterations>')
	 		sys.exit()
		elif opt in ("-b", "--batchsize"):
	 		BATCHSIZE = int(arg)
	 		print(BATCHSIZE)
		elif opt in ("-u", "--n_layers_velocity"):
			N_LAYERS_V = int(arg)
			print(N_LAYERS_V)
		elif opt in ("-p", "--n_layers_pressure"):
			N_LAYERS_P = int(arg)
		elif opt in ("-m", "--max_iterations"):
			print(arg)
			max_iter = int(arg)
		elif opt in ("-r", "--random_seed"):
			seed = int(arg)
			tf.set_random_seed(seed)

	HIDDEN_UNITS_VELOCITY = [] 
	HIDDEN_UNITS_PRESSURE = []

	for i in range(N_LAYERS_V):
		HIDDEN_UNITS_VELOCITY.append(16)

	for i in range(N_LAYERS_P):
		HIDDEN_UNITS_PRESSURE.append(16)

	nu = 0.025

	do_save = True
	save_name = 'test_model/' + str(len(HIDDEN_UNITS)) + '_layer_sq_loss_' + str(BATCHSIZE) + '_m_iter_' + str(max_iter) + '_rs_' + str(seed) 

	problem = kovasznay_flow_problem.kovasznay_flow(nu)
	
	sampler = sampling_from_dataset('datasets/' + str(BATCHSIZE), BATCHSIZE)
	sampler.load_dataset()

	neural_network = neural_networks.neural_networks(2, 2, HIDDEN_UNITS_VELOCITY, HIDDEN_UNITS_PRESSURE, 2, 1)

	velocity_int_var = tf.placeholder(tf.float64, [None, 2]) 
	velocity_init_var = tf.placeholder(tf.float64, [None, 2]) 
	pressure_int_var = tf.placeholder(tf.float64, [None, 2])
	velocity_bou_var = tf.placeholder(tf.float64, [None, 2]) 
	
	velocity_int = neural_network.nn_velocity.value_nn_velocity(velocity_int_var)
	velocity_init = neural_network.nn_velocity.value_nn_velocity(velocity_init_var)
	pressure_int = neural_network.nn_pressure.value_nn_pressure(pressure_int_var)
	velocity_bou = neural_network.nn_velocity.value_nn_velocity(velocity_bou_var)

	grad = neural_network.nn_velocity.first_derivatives_nn_velocity(velocity_int_var)
	grad_grad = neural_network.nn_velocity.second_derivatives_nn_velocity(velocity_int_var)
	grad_p = neural_network.nn_pressure.first_derivates_nn_pressure_multidimensional(pressure_int_var)

	p_x = tf.slice(grad_p[0][0], [0,0], [BATCHSIZE,1])
	p_y = tf.slice(grad_p[0][0], [0,1], [BATCHSIZE,1])

	sol_int_x = tf.placeholder(tf.float64, [None, 1])
	sol_int_y = tf.placeholder(tf.float64, [None, 1])

	sol_bou_x = tf.placeholder(tf.float64, [None, 1])
	sol_bou_y = tf.placeholder(tf.float64, [None, 1])

	u_xx = tf.slice(grad_grad[0][0][0], [0, 0], [BATCHSIZE, 1])
	u_yy = tf.slice(grad_grad[0][1][0], [0, 1], [BATCHSIZE, 1])
	v_xx = tf.slice(grad_grad[1][0][0], [0, 0], [BATCHSIZE, 1])
	v_yy = tf.slice(grad_grad[1][1][0], [0, 1], [BATCHSIZE, 1])

	u_x = tf.slice(grad[0][0], [0, 0], [BATCHSIZE, 1])
	u_y = tf.slice(grad[0][0], [0, 1], [BATCHSIZE, 1])

	v_x = tf.slice(grad[1][0], [0, 0], [BATCHSIZE, 1])
	v_y = tf.slice(grad[1][0], [0, 1], [BATCHSIZE, 1])

	vel_bou_x = tf.slice(velocity_bou, [0, 0], [BATCHSIZE, 1])
	vel_bou_y = tf.slice(velocity_bou, [0, 1], [BATCHSIZE, 1])

	vel_x = tf.slice(velocity_int, [0, 0], [BATCHSIZE, 1])
	vel_y = tf.slice(velocity_int, [0, 1], [BATCHSIZE, 1])

	advection_x = u_x*vel_x + u_y*vel_y
	advection_y = v_x*vel_x + v_y*vel_y

	loss_int = tf.square(-advection_x+nu*(u_xx+u_yy)+sol_int_x-p_x) + tf.square(-advection_y+nu*(v_xx+v_yy)+sol_int_y-p_y)
	loss_div = tf.square(u_x+v_y)
	loss_bou_x = tf.square(vel_bou_x-sol_bou_x)
	loss_bou_y = tf.square(vel_bou_y-sol_bou_y)
	loss_bou = loss_bou_x + loss_bou_y

	loss = tf.sqrt(tf.reduce_mean(INT_WEIGHT*loss_int + BOU_WEIGHT*loss_bou + DIV_WEIGHT*loss_div))

	train_scipy = tf.contrib.opt.ScipyOptimizerInterface(loss, method='BFGS', options={'gtol':1e-14, 'disp':True, 'maxiter':max_iter})

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()

	f_x = np.zeros(BATCHSIZE)
	f_y = np.zeros(BATCHSIZE)

	bou_x = np.zeros(BATCHSIZE)
	bou_y = np.zeros(BATCHSIZE)

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

		f_x = problem.rhs(int_draw)[:,0]
		f_y = problem.rhs(int_draw)[:,1]

		f_x = np.reshape(np.array(f_x), (BATCHSIZE, 1))
		f_y = np.reshape(np.array(f_y), (BATCHSIZE, 1))

		bou_x, bou_y = problem.velocity(bou_draw)
		bou_x = np.reshape(np.array(bou_x), (BATCHSIZE, 1))
		bou_y = np.reshape(np.array(bou_y), (BATCHSIZE, 1))

		train_scipy.minimize(sess, feed_dict={sol_int_x:f_x, sol_int_y:f_y, sol_bou_x:bou_x, sol_bou_y:bou_y, velocity_int_var:int_draw, velocity_bou_var:bou_draw, pressure_int_var:int_draw})
				
		if do_save:		
			save_path = saver.save(sess, save_name)
			print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
	main(sys.argv[1:])
