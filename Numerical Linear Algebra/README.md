# About
SkolTech course

Numerical linear algebra forms the basis for all modern computational mathematics. It is not possible to develop new large scale algorithms and even use existing ones without knowing it. In this course I will show, how numerical linear algebra methods and algorithms are used to solve practical problems. Matrix decompositions play the key role in numerical linear algebra. We will study different matrix decompositions in details: what are they, how to compute them efficiently and robustly, and most importantly, how they are applied to the solution of linear systems, eigenvalue problems and data analysis applications. For large-scale problems iterative methods will be described. I will try to highlight recent developments when it is relevant to the current lecture. This course should serve as a basis for other IT Skoltech courses. It will also serve as a first-time place where programming environment and infrastructure is introduced in a consistent manner.

# Shedule
<p align="center">
  <img src="Shedule NLA.png" >
</p>

# Assignments

- 1.1 Implement Convolution/Deconvolution of signal 
- 1.2 Theoretical tasks
- 1.3 Implement Strassen algorithm
- 1.4 SVD in image compression

- 2.1 LU decomposition
- 2.2 QR decomposition
- 2.3 Word2Vec as Matrix Factorization
- 2.4 Eigenvalues

- 3.1 Theoretical tasks (columnwise/rownwise reshape, estimate analytically the number of iterations [Richardson iteration with the optimal choice of parameter (use  2 -norm), Chebyshev iteration (use  2 -norm), Conjugate gradient method (use  𝐴 -norm).] required to solve linear system with the relative accuracy 
- 3.2 Spectral graph partitioning and inverse iteration (Inverse power method, Spectral graph properties, Image bipartition )
- 3.3 Toy task: You received a radar-made air scan data of a terrorist hideout made from a heavy-class surveillance drone. Unfortunately, it was made with an old-fashioned radar, so the picture is convolved with the diffractive pattern. You need to deconvolve the picture to recover the building plan

# Project 
Quantum tomography (QT) is applied on a source of systems, to determine what the quantum state is of the output of that source. Unlike a measurement on a single system, which determines the system’s current state after the measurement (in general, the act of making a measurement alters the quantum state), quantum tomography works to determine the state prior to the measurements. It is used in quantum computing to examine the result of quantum algorithm (i.e. state of a qubit) or in quantum conmmunications. The computational part of QT requires to solve system of linear equations. 
What makes NLA algorithms useful for this problem.

During this project our team consider the following tasks:
- Simulate an experiment of a quantum state measurement;
- Apply different approaches to the problem of QST, such that pseudo-inverse matrix, direct gradi-
ent algorithm, hedged likelihood method and semi-definite programming methods;
- Compare mentioned methods in terms of complexity and accuracy;

