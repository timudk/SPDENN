import numpy as np

class kovasznay_flow:
	def __init__(self, nu):
		self.nu = nu
		self.lam = 1/(2*nu) - np.sqrt(1/(4*nu*nu)+4*np.pi*np.pi)
		self.x_range = [-0.5, 1]
		self.y_range = [-0.5, 1.5]

	def velocity(self, x):
		u = np.zeros(x.shape)

		if(x.ndim > 1):
			for i in range(x.shape[0]):
				u[i,0] = 1 - np.exp(self.lam*x[i,0])*np.cos(2*np.pi*x[i,1])
				u[i,1] = (self.lam/(2*np.pi))*np.exp(self.lam*x[i,0])*np.sin(2*np.pi*x[i,1])

			return u[:,0], u[:,1]
		else:
			u[0] = 1 - np.exp(self.lam*x[0])*np.cos(2*np.pi*x[1])
			u[1] = (self.lam/(2*np.pi))*np.exp(self.lam*x[0])*np.sin(2*np.pi*x[1])

			return u[0], u[1]


	def velocity_magnitude(self, x):
		magnitude = np.zeros(x.shape[0])

		for i in range(x.shape[0]):
			u_1, u_2 = self.velocity(np.array([x[i,0], x[i,1]]))
			magnitude[i] = u_1**2 + u_2**2

		return magnitude

	def pressure(self, x):

		if (x.ndim==1):
			p = 0.5*(1-np.exp(2*self.lam*x[0]))
		else:
			p = np.zeros(x.shape[0])
			for i in range(x.shape[0]):
				p[i] = 0.5*(1-np.exp(2*self.lam*x[i,0]))

		return p

	def rhs(self, x):
		return np.zeros(x.shape)

