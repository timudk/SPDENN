# A Dicussion on Solving Partial Differential Equations using Neural Networks
Can neural networks learn to solve partial differential equations (PDEs)?  We investigate this question for two (systems of) PDEs, namely, the Poisson equation and the steady Navier–Stokes equations.

* Tim Dockhorn. "A Discussion on Solving Partial Differential Equations using Neural Networks." arXiv preprint

## Poisson problem
The Poisson problem under consideration is given as 
$$ \begin{align}
\begin{split}
    -\nabla^2 u (\bm{x}) &= f(\bm{x}) \quad \text{in } \Omega, \\
    u (\bm{x}) &= g(\bm{x}) \quad \text{on } \partial \Omega,
\end{split}
\end{align} $$

## Setup
We recommend the following package versions to reproduce the results of the paper
* Tensorflow: 1.12.0
* Numpy: 1.16.1
* Scipy: 1.2.0
* Matplotlib: 3.0.2
