import tensorflow as tf
import numpy as np 
import poisson_problem
import neural_networks
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as ticker

rc('font', **{'size':12, 'family':'serif', 'serif':['Computer Modern Roman']})
rc('text', usetex=True)


def draw_magnitude_int_loss(x_range, y_range, session, x_qual, y_qual):
	x = np.linspace(x_range[0], x_range[1], x_qual)
	y = np.linspace(y_range[0], y_range[1], y_qual)

	mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)

	f = problem.rhs(mesh)

	loss_int_magnitude = np.sqrt(session.run(loss_int, feed_dict={int_var: mesh, sol_int:f}))

	return np.reshape(loss_int_magnitude, (x_qual, y_qual))


def draw_magnitude_of_err_2d(x_range, y_range, exact_sol, x_qual, y_qual, neural_net_sol):
	x = np.linspace(x_range[0], x_range[1], x_qual)
	y = np.linspace(y_range[0], y_range[1], y_qual)


	mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
	u_sol = exact_sol(mesh)

	neural_net_sol_mesh = neural_net_sol(mesh.astype(np.float64)).eval()


	err_vec = np.zeros(x_qual*y_qual)

	for i in range(x_qual*y_qual):
		err_vec[i] = np.sqrt((u_sol[i]-neural_net_sol_mesh[i])**2)

	err_vec = np.reshape(err_vec, (x_qual, y_qual))
	return err_vec

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

NUM_STEPS = 1
NUM_INPUTS = 2
BATCHSIZE = 101*101 
HIDDEN_UNITS = [16]

restore_name = 'test_model/1_layer_sq_loss_4000m_iter_20000'
problem = poisson_problem.poisson_2d()

neural_network = neural_networks.neural_network(NUM_INPUTS, 1, HIDDEN_UNITS)

int_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 
bou_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 

value_int = neural_network.value(int_var)
value_bou = neural_network.value(bou_var)

grad = neural_network.first_derivatives(int_var)
grad_grad= neural_network.second_derivatives(int_var)

sol_int = tf.placeholder(tf.float64, [None, 1])
sol_bou = tf.placeholder(tf.float64, [None, 1])

loss_int = tf.square(grad_grad[0]+grad_grad[1]+sol_int)
loss_bou = tf.square(value_bou-sol_bou)
loss = tf.reduce_mean(loss_int + loss_bou)


init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	saver.restore(sess, restore_name)
	print("Model restored.")

	err = draw_magnitude_of_err_2d(problem.range, problem.range, problem.velocity, 101, 101, neural_network.value)
	plt.imshow(np.rot90(err), cmap='hot', interpolation='nearest', extent=[0.0,1.0,0.0,1.0], aspect='auto')
	plt.xlabel(r'$x_1$')
	plt.ylabel(r'$x_2$')
	plt.colorbar(format=ticker.FuncFormatter(fmt))
	plt.show()

	err = draw_magnitude_int_loss(problem.range, problem.range, sess, 101, 101)
	plt.imshow(np.rot90(err), cmap='hot', interpolation='nearest', extent=[0.0,1.0,0.0,1.0], aspect='auto')
	plt.xlabel(r'$x_1$')
	plt.ylabel(r'$x_2$')
	plt.colorbar(format=ticker.FuncFormatter(fmt))
	plt.show()




				
		