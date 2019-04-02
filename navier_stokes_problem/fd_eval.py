import tensorflow as tf
import numpy as np 
import kovasznay_flow_problem
import neural_networks
import matplotlib.pyplot as plt
from matplotlib import rc
import sys, getopt

# python3 fd_eval.py -u 2 -p 1 -f test_model/v_2_p_1_layer_sq_loss_8000_float_64 -x 151 -y 201

def main(argv):
	try:
		opts, args = getopt.getopt(argv,"hu:p:f:x:y:",["n_layers_velocity=", "n_layers_pressure=", "filename=", "N_points_x=", "N_points_y="])
	except getopt.GetoptError:
		print('poisson.py -u <n_layers_velocity> -p <n_layers_pressure> -f <filename> -x <N_points_x> -y <N_points_y>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
	 		print('poisson.py -u <n_layers_velocity> -p <n_layers_pressure> -f <filename> -x <N_points_x> -y <N_points_y>')
	 		sys.exit()
		elif opt in ("-u", "--n_layers_velocity"):
			N_LAYERS_V = int(arg)
		elif opt in ("-p", "--n_layers_pressure"):
			N_LAYERS_P = int(arg)
		elif opt in ("-f", "--filename"):
			restore_name = arg
		elif opt in ("-x", "--N_points_x"):
			N_x = int(arg)
		elif opt in ("-y", "--N_points_x"):
			N_y = int(arg)

	NUM_STEPS = 1
	BATCHSIZE = N_x*N_y

	HIDDEN_UNITS_VELOCITY = [] 
	HIDDEN_UNITS_PRESSURE = []

	for i in range(N_LAYERS_V):
		HIDDEN_UNITS_VELOCITY.append(16)

	for i in range(N_LAYERS_P):
		HIDDEN_UNITS_PRESSURE.append(16)

	nu = 0.025

	problem = kovasznay_flow_problem.kovasznay_flow(nu)

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

	loss = tf.reduce_mean(loss_int + loss_bou + loss_div)

	train_scipy = tf.contrib.opt.ScipyOptimizerInterface(loss, method='BFGS', options={'gtol':1e-14, 'disp':True, 'maxiter':5000})

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)

		saver.restore(sess, restore_name)
		
		def compute_fd_error(session, problem, N_x, N_y):
			x = np.linspace(problem.x_range[0], problem.x_range[1], N_x)
			y = np.linspace(problem.y_range[0], problem.y_range[1], N_y)

			mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
			vel_x, vel_y = problem.velocity(mesh)
			vel_x = np.reshape(np.array(vel_x), (mesh.shape[0], 1))
			vel_y = np.reshape(np.array(vel_y), (mesh.shape[0], 1))

			velo = session.run(velocity_int, feed_dict={velocity_int_var: mesh})
			fd_error_u_1 = (np.reshape(velo[:,0], (mesh.shape[0], 1)) - vel_x)**2
			fd_error_u_2 = (np.reshape(velo[:,1], (mesh.shape[0], 1)) - vel_y)**2

			sum_fd_error_u_1 = np.sum(fd_error_u_1)
			sum_fd_error_u_2 = np.sum(fd_error_u_2)

			return sum_fd_error_u_1/(mesh.shape[0]), sum_fd_error_u_2/(mesh.shape[0]), np.max(fd_error_u_1), np.max(fd_error_u_2)


		def compute_fd_loss(session, problem, N_x, N_y):
			x = np.linspace(problem.x_range[0], problem.x_range[1], N_x)
			y = np.linspace(problem.y_range[0], problem.y_range[1], N_y)

			mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)

			f_x = problem.rhs(mesh)[:,0]
			f_y = problem.rhs(mesh)[:,1]

			f_x = np.reshape(np.array(f_x), (mesh.shape[0], 1))
			f_y = np.reshape(np.array(f_y), (mesh.shape[0], 1))

			fd_loss_int = session.run(loss_int, feed_dict={velocity_int_var: mesh, pressure_int_var: mesh, sol_int_x: f_x, sol_int_y: f_y,})
			fd_loss_div = session.run(loss_div, feed_dict={velocity_int_var: mesh})

			sum_fd_loss_int = np.sum(fd_loss_int)
			sum_fd_loss_div = np.sum(fd_loss_div)

			return sum_fd_loss_int/(mesh.shape[0]), sum_fd_loss_div/(mesh.shape[0]), np.max(fd_loss_int), np.max(fd_loss_div)


		u_1_error, u_2_error, u_1_error_max, u_2_error_max = compute_fd_error(sess, problem, N_x, N_y)
		loss_int, loss_div, loss_int_max, loss_div_max = compute_fd_loss(sess, problem, N_x, N_y)

		print(np.sqrt(u_1_error), np.sqrt(u_2_error), np.sqrt(u_1_error_max), np.sqrt(u_2_error_max), np.sqrt(loss_int), np.sqrt(loss_div), np.sqrt(loss_int_max), np.sqrt(loss_div_max))


if __name__ == '__main__':
	main(sys.argv[1:])


				
		