import poisson_problem
import numpy as np
import tensorflow as tf
import neural_networks
import sys, getopt

# python3 fd_eval.py -n 1 -f test_model/1_layer_sq_loss_4000m_iter_20000 -N 101

def main(argv):

	try:
		opts, args = getopt.getopt(argv,"hn:f:N:",["n_layers=", "filename=", "N_points_per_dim="])
	except getopt.GetoptError:
		print('poisson.py -n <n_layers> -f <filename> -N <N_points_per_dim>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
	 		print('poisson.py -n <n_layers> -f <filename> -N <N_points_per_dim>')
	 		sys.exit()
		elif opt in ("-n", "--n_layers"):
			N_LAYERS = int(arg)
		elif opt in ("-f", "--filename"):
			FILENAME = arg
		elif opt in ("-N", "--N_points_per_dim"):
			N = int(arg)

	
	PROBLEM = poisson_problem.poisson_2d()

	NUM_INPUTS = 2
	NUM_OUTPUTS = 1
	HIDDEN_UNITS = []
	for i in range(N_LAYERS):
		HIDDEN_UNITS.append(16)

	neural_network = neural_networks.neural_network(NUM_INPUTS, NUM_OUTPUTS, HIDDEN_UNITS)

	int_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 
	bou_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 

	value_int = neural_network.value(int_var)
	value_bou = neural_network.value(bou_var)

	grad = neural_network.first_derivatives(int_var)
	grad_grad= neural_network.second_derivatives(int_var)

	sol_int = tf.placeholder(tf.float64, [None, 1])
	sol_bou = tf.placeholder(tf.float64, [None, 1])

	sum_of_second_derivatives = 0.0
	for i in range(NUM_INPUTS):
		sum_of_second_derivatives += grad_grad[i]

	loss_int = tf.square(sum_of_second_derivatives + sol_int)
	loss_bou = tf.square(value_bou-sol_bou)

	loss = tf.reduce_mean(loss_int + loss_bou)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	def compute_fd_error(session, problem, N):
		x = np.linspace(problem.range[0], problem.range[1], N)
		y = np.linspace(problem.range[0], problem.range[1], N)

		mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
		fd_error = (session.run(value_int, feed_dict={int_var: mesh})- problem.velocity(mesh))**2

		sum_fd_error = np.sum(fd_error)

		return sum_fd_error/(N**2)


	def compute_fd_loss(session, problem, N):
		x = np.linspace(problem.range[0], problem.range[1], N)
		y = np.linspace(problem.range[0], problem.range[1], N)

		mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
		sol = np.reshape(problem.rhs(mesh), ((N*N), 1))
		fd_loss = session.run(loss_int, feed_dict={int_var: mesh, sol_int: sol})

		sum_fd_loss = np.sum(fd_loss)

		return sum_fd_loss/(N**2)

	def compute_fd_error_max(session, problem, N):
		x = np.linspace(problem.range[0], problem.range[1], N)
		y = np.linspace(problem.range[0], problem.range[1], N)

		mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
		fd_error = (session.run(value_int, feed_dict={int_var: mesh})- problem.velocity(mesh))**2

		arg_max = np.argmax(fd_error)

		return fd_error[arg_max], mesh[arg_max]


	def compute_fd_loss_max(session, problem, N):
		x = np.linspace(problem.range[0], problem.range[1], N)
		y = np.linspace(problem.range[0], problem.range[1], N)

		mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
		sol = np.reshape(problem.rhs(mesh), ((N*N), 1))
		fd_loss = session.run(loss_int, feed_dict={int_var: mesh, sol_int: sol})

		arg_max = np.argmax(fd_loss)

		return fd_loss[arg_max], mesh[arg_max]

	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess, FILENAME)
		print('Model restored.')

		print('l2-loss:', np.sqrt(compute_fd_error(sess, PROBLEM, N)))
		print('l2-int-loss:', np.sqrt(compute_fd_loss(sess, PROBLEM, N)))

		l_2_max, l_2_max_location = compute_fd_error_max(sess, PROBLEM, N)
		l_2_int_max, l_2_int_max_location = compute_fd_loss_max(sess, PROBLEM, N)

		print('l2-max:', np.sqrt(l_2_max), l_2_max_location)
		print('l2-int-max:', np.sqrt(l_2_int_max), l_2_int_max_location)

if __name__ == '__main__':
	main(sys.argv[1:])





