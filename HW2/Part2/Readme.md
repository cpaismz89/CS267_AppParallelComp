# Assignment 2: Parallelizing a Particle Simulation
## Problem Statement
This assignment is an introduction to parallel programming using shared memory and distributed memory programming models.
In this assignment, we will be parallelizing a toy particle simulation (similar simulations are used in mechanics, biology, and astronomy).  In our simulation, particles interact by repelling one another.  
The particles repel one another, but only when closer than a cutoff distance highlighted around one particle in grey.

## Part 2: MPI
There is one executable (mpi.cpp) you need to submit, along with a short report. You need to create one distributed memory implementation (MPI) that runs in O(n) time with hopefully O(n/p) scaling.