#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <stdio.h>
#include <cuda.h>

namespace matrixMapp {
    class matrixCells;
}

/**
 * Size of the particle simulation area's sides.
 */
extern double size;

//
//  tuned constants
//
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005


inline int mymin( int a, int b ) { return a < b ? a : b; }
inline int mymax( int a, int b ) { return a > b ? a : b; }

//
//  saving parameters
//
const int NSTEPS = 1000;
const int SAVEFREQ = 10;

//
// particle data structure
//
typedef struct
{
    double x;
    double y;
    double vx;
    double vy;
    double ax;
    double ay;
} particle_t;

//
//  timing routines
//
double read_timer( );

//
//  simulation routines
//
void set_size( int n );
void init_particles( int n, particle_t p[] );
void push2Mesh(int n, particle_t p[], matrixMapp::matrixCells*);
void apply_force( particle_t &particle, particle_t &neighbor , double *dmin, double *davg, int *navg);
void move( particle_t &p );

//
// CUDA routines (GPU)
//
__device__ int IDparticle(double x, double y, double meshSize, int xmesh);
__device__ int IDparticle(particle_t &particle, double meshSize, int xmesh);
__global__ void push2mesh_gpu(particle_t * particles, int n, int* adj,
                              int* submesh, double meshSize, int xmesh);
__global__ void clear(int Nmeshs, int* submesh);
__device__ void apply_force_gpu(particle_t &particle, particle_t &adjCell);
__device__ void submeshForce(particle_t * particles, int tid, int * adj, int submesh);
__device__ void submeshForceAll(particle_t * particles, int tid, int * adj, int submesh);
__global__ void compute_forces_gpu(particle_t * particles, int n, int * adj,
                                   int * submesh, double meshSize, int xmesh);
__global__ void compute_forces_mesh_gpu(particle_t * particles, int * adj,
                                        int Nmeshs, int * submesh, double meshSize, int xmesh);
__global__ void move_gpu (particle_t * particles, int n, double size);

//
//  I/O routines
//
FILE *open_save( char *filename, int n );
void save( FILE *f, int n, particle_t *p );

//
//  argument processing routines
//
int find_option( int argc, char **argv, const char *option );
int read_int( int argc, char **argv, const char *option, int default_value );
char *read_string( int argc, char **argv, const char *option, char *default_value );

#endif
