// Inclusions
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// CUDA NVIDIA lib
#include <cuda.h>

// Common lib
#include "common_gpu.h"

// Adapted mesh for GPU
//#include <matrixCells_gpu.h>

// Number of threads (as in Vanilla)
#define NUM_THREADS 256

// Auxiliary constant for defining mesh dimensions (for 1 axis)
const double meshDim = 2.0 * cutoff;

// Auxiliary parameters
int xmesh, Nmeshs;
double meshSize;

// Size initialization
extern double size;

//
//  benchmarking program
//


//////////////////////////////////////////////////////////////////
//
//                             Main program
//
//////////////////////////////////////////////////////////////////
int main( int argc, char **argv )
{
    //////////////////////////////////////////////////////////////////
    //
    //                    Same as Vanilla section
    //
    //////////////////////////////////////////////////////////////////

    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    // Total number of particles
    int n = read_int( argc, argv, "-n", 1000 );

    // Name of the output file
    char *savename = read_string( argc, argv, "-o", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    // Number of particles n
    set_size( n );

    // Initialize particles (not optimized)
    init_particles( n, particles );

    // Synchronization
    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    // Synchronization
    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    //////////////////////////////////////////////////////////////////
    //
    //                    End same as Vanilla section
    //
    //////////////////////////////////////////////////////////////////



    //////////////////////////////////////////////////////////////////
    //
    //                    Meshs are initialized
    //
    //////////////////////////////////////////////////////////////////

    // Mesh initialization time stats
    double meshInitTime = read_timer();

    // Auxiliary parameters initialization
    // One direction dimension for the mesh
    xmesh = (int) ceil (size * 1.0 / meshDim);

    // Total number of grids by multiplying xmesh * xmesh
    Nmeshs = xmesh * xmesh;

    // Mesh size
  	meshSize = size / xmesh;

  	// Compute the number of blocks based on the number of threads
  	int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;

  	// Compute the number of "mesh blocks" based on the number of threads
    int meshBlocks = (Nmeshs + NUM_THREADS - 1) / NUM_THREADS;

    // Initialize submesh and adj pointers and allocate memory
    int * submesh;
    cudaMalloc((void **) &submesh, Nmeshs * sizeof(int));
    int * adj;
    cudaMalloc((void **) &adj, n * sizeof(int));

    // Synchronization step
    cudaThreadSynchronize();

    // Clear the mesh multi-threaded: Kernel invocation with NUM_THREADS threads
    // From NVIDIA: Here, each of the N threads that execute
    //              clear, do that with the submeshs
    clear <<< meshBlocks, NUM_THREADS >>> (Nmeshs, submesh);

    // Assign the particles multi-threaded: particles are assigned to
    push2mesh_gpu <<< blocks, NUM_THREADS >>> (d_particles, n, adj, submesh, meshSize, xmesh);

    cudaThreadSynchronize();

    // Calculate the total amount of time spent creating the underlying data structure
    meshInitTime = read_timer() - meshInitTime;


    //////////////////////////////////////////////////////////////////
    //
    //                       End Mesh section
    //
    //////////////////////////////////////////////////////////////////


    //
    //  simulate a number of time steps (Vanilla)
    //
    // Synchronization step
    cudaThreadSynchronize();

    // Init simulation timer
    double simulation_time = read_timer( );

    // Main loop
    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //
        // Compute forces multi-threaded: forces are computed using the mesh structure
        compute_forces_gpu <<< blocks, NUM_THREADS >>> (d_particles, n, adj, submesh, meshSize, xmesh);


        //
        //  move particles (Vanilla)
        //
        move_gpu <<< blocks, NUM_THREADS >>> (d_particles, n, size);


       // Update the particles inside the mesh
       // Clear the meshs (multi-threaded)
       clear <<< meshBlocks, NUM_THREADS >>> (Nmeshs, submesh);

       // Push particles to meshs: multi-threaded
       push2mesh_gpu <<< blocks, NUM_THREADS >>> (d_particles, n, adj, submesh, meshSize, xmesh);

        //
        //  save if necessary (Vanilla)
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
        }
    }
    // Synchronization step and compute total simulation time (as in Vanilla)
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    // Print information about the simulation and GPU (Vanilla + mesh info)
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "GPU mesh initialization time = %g seconds\n", meshInitTime);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    // Release resources (Vanilla)
    free( particles );
    cudaFree(d_particles);

    // Specifics from mesh approach
    cudaFree(submesh);
    cudaFree(adj);

    // Close file if open
    if( fsave )
        fclose( fsave );

    // End of the main
    return 0;
}
