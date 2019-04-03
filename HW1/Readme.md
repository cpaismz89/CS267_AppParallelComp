# Assignment 1: Optimizing Matrix Multiplication
## Problem statement
Your task in this assignment is to write an optimized matrix multiplication function for NERSC's Cori supercomputer.  We will give you a generic matrix multiplication code (also called matmul or dgemm), and it will be your job to tune our code to run efficiently on Cori's processors.  There are two parts to this assignment.

## Part One
- Write an optimized single-threaded matrix multiply kernel.  This will run on only one core.

## Part Two
- Write an optimized multi-threaded matrix multiply kernel.  This will run on one processor, using all the available cores.

We consider a special case of matmul:
C := C + A*B

# Authors
- Cristobal Pais
- Alexander Wu
- Yucong He 