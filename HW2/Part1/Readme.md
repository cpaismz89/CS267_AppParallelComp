# Assignment 2: Parallelizing a Particle Simulation
## Problem Statement
This assignment is an introduction to parallel programming using shared memory and distributed memory programming models.
In this assignment, we will be parallelizing a toy particle simulation (similar simulations are used in mechanics, biology, and astronomy).  In our simulation, particles interact by repelling one another.  
The particles repel one another, but only when closer than a cutoff distance highlighted around one particle in grey.

## Part 1: Part 1: Serial and OpenMP
In this assignment, you will write two versions of our simulation.  First, you will write an O(n) serial code.  Then, you will write a parallel version of this O(n) code for shared memory architectures using OpenMP.
There are two executables and one short report you need to submit. You need to create at a minimum one serial code (serial.cpp) that runs in O(n) time and one shared memory implementation (openmp.cpp) using OpenMP.