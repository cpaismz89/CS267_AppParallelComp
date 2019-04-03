# Assignment 2: Parallelizing a Particle Simulation
## Problem Statement
This assignment is an introduction to parallel programming using shared memory and distributed memory programming models.
In this assignment, we will be parallelizing a toy particle simulation (similar simulations are used in mechanics, biology, and astronomy).  In our simulation, particles interact by repelling one another.  
The particles repel one another, but only when closer than a cutoff distance highlighted around one particle in grey.

## Part 3: GPU
We have provided a naive $O(n^2)$ GPU implementation, similar to the openmp, and MPI codes listed above. It will be your task to make the necessary algorithmic changes to obtain an O(n) GPU implementation and other machine optimizations to achieve favorable performance across a range of problem sizes.
A simple correctness check which computes the minimal distance between 2 particles during the entire simulation is provided.  A correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between 0.7-0.8.  A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between 0.01-0.05 . 
Adding the checks inside the GPU code provides too much of an overhead so an autocorrect executable is provided that checks the output txt file for the values mentioned above.  We also provide a clean, $O(n^2)$ serial implementation, serial.cu.