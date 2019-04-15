# A Dicussion on Solving Partial Differential Equations using Neural Networks
Can neural networks learn to solve partial differential equations (PDEs)?  We investigate this question for two (systems of) PDEs, namely, the Poisson equation and the steady Navier–Stokes equations.

* Tim Dockhorn. "A Discussion on Solving Partial Differential Equations using Neural Networks." arXiv preprint

## Poisson problem
A neural network (with two fully connected layers of size 16) for the manufactured Poisson problem (using dataset of 2000 interior and boundary points) can be trained using the following command:
```console
foo@bar:~$ python3 poisson.py -b 2000 -n 2
```

## Navier--Stokes problem
A velocity and a pressure neural network (with two fully connected layers size 16 each) for the Kovasznay problem (using dataset of 4000 interior and boundary points) can be trained using the following command:
```console
foo@bar:~$ python3 kovasznay_flow.py -b 4000 -u 2 -p 2
```

## Setup
We recommend the following package versions to reproduce the results of the paper
* Tensorflow: 1.12.0
* Numpy: 1.16.1
* Scipy: 1.2.0
* Matplotlib: 3.0.2
