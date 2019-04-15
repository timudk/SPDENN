import numpy as np
import csv
import poisson_problem

N = 16000

class sampling_from_rectangle:
	def __init__(self, x_range, y_range):
		self.x_range = x_range
		self.y_range = y_range

	def interior_samples(self, batchsize):
		int_draw_x = np.random.uniform(self.x_range[0], self.x_range[1]+self.epsilon, batchsize)
		int_draw_y = np.random.uniform(self.y_range[0], self.y_range[1]+self.epsilon, batchsize)

		return int_draw_x, int_draw_y

	def boundary_samples(self, batchsize):
		a = self.x_range[1]-self.x_range[0]
		b = self.y_range[1]-self.y_range[0]

		draw_perimeter = np.random.uniform(0, 2*(a + b), batchsize)

		draw = []

		for i in draw_perimeter:
			if i < a:
				draw.append([i+ self.x_range[0], self.y_range[0]])
			elif a <= i and i < a+b:
				draw.append([self.x_range[1], (i-a) + self.y_range[0]])
			elif a+b <= i and i < 2*a+b:
				draw.append([self.x_range[1] - (i-(a+b)), self.y_range[1]])
			elif 2*a+b <= i and i<= 2*a+2*b:
				draw.append([self.x_range[0], self.y_range[1] - (i-(2*a+b))])

		return np.array(draw)[:, 0], np.array(draw)[:, 1]

problem = poisson_problem.poisson_2d()
sampler = sampling_from_rectangle(problem.range, problem.range)

filename = 'datasets/'

def main():
	with open(filename + str(N) + extension, mode='w') as f:
		csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		int_draw_x, int_draw_y = sampler.interior_samples(N)
		bou_draw_x, bou_draw_y= sampler.boundary_samples(N)
		
		for i in range(N):
			csv_writer.writerow([int_draw_x[i], int_draw_y[i], bou_draw_x[i], bou_draw_y[i]])

if __name__ == '__main__':
	main()
