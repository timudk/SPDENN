import tensorflow as tf
import numpy as np 
import kovasznay_flow_problem
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

	f_x = problem.rhs(mesh)[:,0]
	f_y = problem.rhs(mesh)[:,1]

	f_x = np.reshape(np.array(f_x), (x_qual*y_qual, 1))
	f_y = np.reshape(np.array(f_y), (x_qual*y_qual, 1))

	loss_int_magnitude = np.sqrt(session.run(loss_int, feed_dict={velocity_int_var: mesh, pressure_int_var: mesh, sol_int_x:f_x, sol_int_y:f_y}))
	loss_div_magnitude = np.sqrt(session.run(loss_div, feed_dict={velocity_int_var: mesh, pressure_int_var: mesh, sol_int_x:f_x, sol_int_y:f_y}))

	return np.reshape(loss_int_magnitude+loss_div_magnitude, (x_qual, y_qual))


def draw_magnitude_of_err_2d(x_range, y_range, exact_sol, x_qual, y_qual, neural_net_sol, type):
	x = np.linspace(x_range[0], x_range[1], x_qual)
	y = np.linspace(y_range[0], y_range[1], y_qual)

	mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)

	if(type=='velocity'):
		u_sol, v_sol = exact_sol(mesh)
	elif(type=='pressure'):
		p_sol = exact_sol(mesh)

	neural_net_sol_mesh = neural_net_sol(mesh.astype(np.float64)).eval()

	err_vec = np.zeros(x_qual*y_qual)

	if(type=='velocity'):
		for i in range(x_qual*y_qual):
			err_vec[i] = np.sqrt((u_sol[i]-neural_net_sol_mesh[i][0])**2) + np.sqrt((v_sol[i]-neural_net_sol_mesh[i][1])**2)

	elif(type=='pressure'):
		for i in range(x_qual*y_qual):
			err_vec[i] = (p_sol[i]-neural_net_sol_mesh[i])**2

	err_vec_image = np.zeros((x_qual, y_qual))
	for i in range(x_qual):
		err_vec_image[i,:] = err_vec[i*y_qual:(i+1)*y_qual] 

	return err_vec_image

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

NUM_STEPS = 1
BATCHSIZE = 30351 

HIDDEN_UNITS_VELOCITY = [16, 16]
HIDDEN_UNITS_PRESSURE = [16] 

nu = 0.025

restore_name = 'test_model/v_2_p_1_layer_sq_loss_8000_float_64'
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

f_x = np.zeros(BATCHSIZE)
f_y = np.zeros(BATCHSIZE)

bou_x = np.zeros(BATCHSIZE)
bou_y = np.zeros(BATCHSIZE)

print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


with tf.Session() as sess:
	sess.run(init)

	saver.restore(sess, restore_name)
	print("Model restored.")

	
	err = draw_magnitude_of_err_2d(problem.x_range, problem.y_range, problem.velocity, 151, 201, neural_network.nn_velocity.value_nn_velocity, 'velocity')
	plt.imshow(np.rot90(err), cmap='hot', interpolation='nearest', extent=[-0.5,1.0,-0.5,1.5])
	plt.xlabel(r'$x_1$')
	plt.ylabel(r'$x_2$')
	plt.colorbar(format=ticker.FuncFormatter(fmt))
	plt.show()


	err = draw_magnitude_int_loss(problem.x_range, problem.y_range, sess, 151, 201)
	plt.imshow(np.rot90(err), cmap='hot', interpolation='nearest')
	plt.xlabel(r'$x_1$')
	plt.ylabel(r'$x_2$')
	plt.colorbar(format=ticker.FuncFormatter(fmt))
	plt.show()

	