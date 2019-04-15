import numpy as np

class poisson_2d:
	def __init__(self):
		self.range = [0.0, 1.0]

	def velocity(self, x):
		u = np.zeros((x.shape[0], 1))

		for i in range(x.shape[0]):
			u[i] = 1.0
			for j in range(2):
				u[i] *= np.sin(np.pi*x[i,j])

		return u

	def rhs(self, x):
		f = np.zeros((x.shape[0], 1))

		for i in range(x.shape[0]):
			f[i] = 2.0*np.pi*np.pi
			for j in range(2):
				f[i] *= np.sin(np.pi*x[i,j])

		return f
