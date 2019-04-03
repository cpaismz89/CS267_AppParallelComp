#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "common.h"
#include <cuda.h>

double size;

//
//  tuned constants
//

//
//  timer
//
double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//
//  keep density constant
//
void set_size( int n )
{
    size = sqrt( density * n );
}

//
//  Initialize the particle positions and velocities
//
void init_particles( int n, particle_t *p )
{
    srand48( time( NULL ) );

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;

    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;

    for( int i = 0; i < n; i++ )
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];

        //
        //  distribute particles evenly to ensure proper spacing
        //
        p[i].x = size*(1.+(k%sx))/(1+sx);
        p[i].y = size*(1.+(k/sx))/(1+sy);

        //
        //  assign random velocities within a bound
        //
        p[i].vx = drand48()*2-1;
        p[i].vy = drand48()*2-1;
    }
    free( shuffle );
}

//
//  interact two particles
//
void apply_force( particle_t &particle, particle_t &neighbor )
{

    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
    r2 = fmax( r2, min_r*min_r );
    double r = sqrt( r2 );

    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

//
//  integrate the ODE
//
void move( particle_t &p )
{
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;

    //
    //  bounce from walls
    //
    while( p.x < 0 || p.x > size )
    {
        p.x  = p.x < 0 ? -p.x : 2*size-p.x;
        p.vx = -p.vx;
    }
    while( p.y < 0 || p.y > size )
    {
        p.y  = p.y < 0 ? -p.y : 2*size-p.y;
        p.vy = -p.vy;
    }
}

//
// CUDA Routines (GPU)
//
// Taking the location of the particle x,y, we transform it into IDs
__device__ int IDparticle(double x, double y, double meshSize, int xmesh) {
	// Coordinates (indexes)
	int xid = x / meshSize;
	int yid = y / meshSize;

	// Return ID location
	return xid * xmesh + yid;
}

// Taking the location of the particle x,y, we transform it into IDs
__device__ int IDparticle(particle_t &particle, double meshSize, int xmesh) {
	// Coordinates (indexes)
	int xid = particle.x / meshSize;
	int yid = particle.y / meshSize;

	// Return ID location
	return xid * xmesh + yid;
}

// Push particles into the mesh
__global__ void push2mesh_gpu(particle_t * particles, int n, int* adj,
                              int* submesh, double meshSize, int xmesh) {
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // If thread is n, stop
  if(tid >= n) return;

  // Otherwise, get next particle ID
  int k = IDparticle(particles[tid], meshSize, xmesh);

  // From NVIDIA: atomicExch(int *address, int val)
  // reads the 32-bit or 64-bit word old located at the address "address" in global or shared memory
  // and stores val back to memory at the same address. These two operations are performed in one
  // atomic transaction.

  // Thread ID is associated with kth particle's sub mesh
  adj[tid] = atomicExch(&submesh[k], tid);
}

// Clear the mesh
__global__ void clear(int Nmeshs, int* submesh) {
  // Get thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // If thread is greater or equal than the total number of existing meshs, stop
  if(tid >= Nmeshs) return;

  // Otherwise, set sub mesh to thread -1
  submesh[tid] = -1;
}

// Apply force gpu function for particle's local neighborhood (same as Vanilla)
__device__ void apply_force_gpu(particle_t &particle, particle_t &adjCell)
{

  double dx = adjCell.x - particle.x;
  double dy = adjCell.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

// Apply forces to own thread's mesh
__device__ void submeshForce(particle_t * particles, int tid, int * adj, int submesh) {
  // Pointer to thread's particles
  particle_t* p = &particles[tid];

  // Apply force to particles with i != current thread
  for(int i = submesh; i != -1; i = adj[i]) {
    if(i != tid)
      apply_force_gpu(*p, particles[i]);
  }
}

// Apply forces to thread's mesh
__device__ void submeshForceAll(particle_t * particles, int tid, int * adj, int submesh) {
  // Pointer to thread's particles
  particle_t* p = &particles[tid];

  // Apply force of particles in the mesh
  for(int i = submesh; i != -1; i = adj[i]) {
      apply_force_gpu(*p, particles[i]);
  }
}

// Compute forces gpu: main mod to deal with own particles and adjacent ones
__global__ void compute_forces_gpu(particle_t * particles, int n, int * adj, int * submesh, double meshSize, int xmesh)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  // Calculate ID and location
  particle_t * p = &particles[tid];
  int xid = p -> x / meshSize;
  int yid = p -> y / meshSize;
  int k = xid * xmesh + yid;

  // Set ax and ay = 0
  p->ax = p->ay = 0;

  // Forces are computed (if needed) inside the submesh
  submeshForce(particles, tid, adj, submesh[k]);

  // Forces are computed (if needed) w.r.t. other submeshs
  // Cases: check corresponding submesh
  // Right
  if(xid > 0) {
	  submeshForceAll(particles, tid, adj, submesh[k - xmesh]);
	  if(yid > 0)
	    submeshForceAll(particles, tid, adj, submesh[k - xmesh - 1]);
	  if(yid < xmesh - 1)
	    submeshForceAll(particles, tid, adj, submesh[k - xmesh + 1]);
  }

  // Left
  if(xid < xmesh - 1) {
      submeshForceAll( particles, tid, adj, submesh[k + xmesh]);
	  if(yid > 0)
	    submeshForceAll(particles, tid, adj, submesh[k + xmesh - 1]);
	  if(yid < xmesh - 1)
	    submeshForceAll(particles, tid, adj, submesh[k + xmesh + 1]);
  }

  // Up
  if(yid > 0) submeshForceAll(particles, tid, adj, submesh[k - 1]);

  // Down
  if(yid < xmesh - 1) submeshForceAll(particles, tid, adj, submesh[k + 1]);
}

// Compute forces for meshs gpu: main mod to deal with own particles and adjacent ones
__global__ void compute_forces_mesh_gpu(particle_t * particles, int * adj,int Nmeshs, int * submesh, double meshSize, int xmesh)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= Nmeshs) return;

  // Calculate ID and location
  int xid = tid / xmesh;
  int yid = tid % xmesh;
  int k = tid;

  // Get particles
  for(int i = submesh[tid]; i != -1; i = adj[i]) {
    particle_t * p = &particles[i];

    // Set ax and ay = 0
    p->ax = p->ay = 0;

    // Forces are computed (if needed) inside the submesh
	submeshForce(particles, i, adj, submesh[k]);

    // Forces are computed (if needed) w.r.t. other submeshs
    // Cases: check corresponding submesh
    // Right
    if(xid > 0) {
        submeshForceAll(particles, i, adj, submesh[k - xmesh]);
	    if(yid > 0)
	      submeshForceAll(particles, i, adj, submesh[k - xmesh - 1]);
	    if(yid < xmesh - 1)
	      submeshForceAll(particles, i, adj, submesh[k - xmesh + 1]);
    }

    // Left
    if(xid < xmesh - 1) {
	    submeshForceAll(particles, i, adj, submesh[k + xmesh]);
	    if(yid > 0)
	      submeshForceAll(particles, i, adj, submesh[k + xmesh - 1]);
	    if(yid < xmesh - 1)
	      submeshForceAll(particles, i, adj, submesh[k + xmesh + 1]);
    }

    // Up
    if(yid > 0) submeshForceAll(particles, i, adj, submesh[k - 1]);

	// Down
    if(yid < xmesh - 1) submeshForceAll(particles, i, adj, submesh[k + 1]);
  }
}

// Move particles function (same as Vanilla)
__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}


//
//  I/O routines
//
void save( FILE *f, int n, particle_t *p )
{
    static bool first = true;
    if( first )
    {
        fprintf( f, "%d %g\n", n, size );
        first = false;
    }
    for( int i = 0; i < n; i++ )
        fprintf( f, "%12.10f %12.10f\n", p[i].x, p[i].y );
}

//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}
