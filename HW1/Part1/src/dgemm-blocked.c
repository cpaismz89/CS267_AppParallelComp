// Calling the AVX intrinsics files
#include <emmintrin.h>
#include <immintrin.h>

// Include the standard stdio and declare the description string
#include <stdio.h>
const char* dgemm_desc = "Loop Unrolling/Ordering + AVX 4x4 + Packing + Padding + Blocked memory";

// Size of the block (To tune!)
#define BLOCK_SIZE 128

// Unroll parameters
#define UNROLL_L1 4
#define UNROLL_L2 8
#define UNROLL_L3 2

// Size for B packed (To tune!)
#define lbb 1200

// Macro for minimum between two values
#define min( i, j ) ( (i)<(j) ? (i): (j) )

// Define functions
void square_dgemm(int lda, double* A, double* B, double* C);
void do_block(int toPackB, int lda, int M, int n, int K, double *A, double *B, double *C);
void process_fringes(int lda, int M, int n, int K, double *A, double *B, double* C);
void Calculate_C_Block8x4(int K, double *A, double* B, double *C, int lda, int ldb, int ldc);
void Calculate_C_Block8x3(int K, double *A, double* B, double *C, int lda, int ldb, int ldc);
void Calculate_C_Block8x8(int K, double *A, double* B, double *C, int lda, int ldb, int ldc);
void Calculate_C_Block4x4(int K, double *A, double* B, double *C, int lda, int ldb, int ldc);
void create_A_8xK(int K, double *A, int lda, double *A8k);
void create_A_4xK(int K, double *A, int lda, double *A4k);
void row_major_B(int K, double *B, int ldb, double *BT4);
void row_major_Bx3(int K, double *B, int ldb, double *BT3);

/////////////////////////////////////////////
//
//               Main Functions
//
/////////////////////////////////////////////

// Main function called by the benchmark file
// Loop over K is omitted in this case (dealing with it inside do_block)
void square_dgemm(int lda, double* A, double* B, double* C)
{
    // Initializing auxiliary variables for leading with the edge cases
    int M, N, NB;
    NB = 128;
    // Columns
    for (int j = 0; j < lda; j+=NB){
        // Blocks the columns of the matrix A
        // Correct block dimensions if block "goes off edge of" the matrix
        N = min(NB, lda - j);

        // Rows
        for (int i = 0; i < lda; i+=NB){
            // Blocks the rows of the matrix C
            // Correct block dimensions if block "goes off edge of" the matrix
            M = min(NB, lda - i);

            // Call the do_block function: Pack B if i = 0 (improving performance)
            do_block(i == 0, lda, M, lda, N, A + i + j*lda, B + j, C + i);

          }
    }
}


// Do block function including unrolling
void do_block(int toPackB, int lda, int M, int n, int K, double *A, double *B, double *C)
{
    // Initialize packed matrices arrays
    double Ap[M*K], Bp[BLOCK_SIZE*lbb];

    // Initialize indexes
    int i, j;

    // Main loop unrolled by 4 (default UNROLL_L1) factor over the colums of the C matrix
    for (j = 0; j < n / UNROLL_L1 * UNROLL_L1; j+=UNROLL_L1){

        // Check if we need to pack the B matrix into a row major order depending on the toPackB flag
        if (toPackB) {
            row_major_B(K, B + j*lda, lda, &Bp[j*K]);
        }

            // Inner loop unrolled by 8 (default UNROLL_L2) factor over the rows of the C matrix
            for (i = 0; i < M / UNROLL_L2 * UNROLL_L2; i+=UNROLL_L2){
            //    printf("i = %d\n", i);
                // If we are starting the loop, create a sub-matrix 8xK from A
                if (j == 0){
                    create_A_8xK(K, A + i, lda, &Ap[i*K]);
                }

                // Calculate the corresponding block of the C matrix (the new matrix in powers of 2 (8x4))
                Calculate_C_Block8x4(K, &Ap[i*K], &Bp[j*K], C + i + j*lda, 8, K, lda);
          }
      }

    // Depending on the matrix dimension, several special cases must be covered (fringes)
    // If the rows are not multiples of 8 but columns are of 4, we need to deal with the remaining block
    if (M % 8 !=0 && n % 4 == 0){

        // 4x4
        if (M % 4 == 0){

            // Inner loop unrolled by 8 (default UNROLL_L2) factor over the rows of the C matrix
            for (j = 0; j < n / UNROLL_L1 * UNROLL_L1; j+=UNROLL_L1){
                for (i = 0; i < M / 4 * 4; i+=4){

                // If we are starting the loop, create a sub-matrix 8xK from A
                if (j == 0){
                    create_A_4xK(K, A + i, lda, &Ap[i*K]);
                }

                // Calculate the corresponding block of the C matrix (the new matrix in powers of 2 (8x4))
                Calculate_C_Block4x4(K, &Ap[i*K], &Bp[j*K], C + i + j*lda, 4, K, lda);
                }
          }
        }
        else {
        // Compute the index associated with the row
        int index_row = M / 8 * 8;

        // Call the auxiliary function for dealing with the fringes
        process_fringes(lda, M - index_row, n, K, A + index_row, B, C + index_row);
        }
    }

    // If neither the rows or columns are multiples of 8 and 4, we need to repack B
    else if (M % 8 != 0 && n % 4 != 0){
        int index_row = M / 8 * 8, index_col = n / 4 * 4;

        // If the remaining columns are 3, repack B
        if (n % 4 == 3){
          // Initialize new repacked matrix with dimension n = 3
          double Bp3[BLOCK_SIZE*3];
          row_major_Bx3(K, B + index_col*lda, lda, &Bp3[0]);

          // Inner loop unrolled by 8 (default UNROLL_L2) factor over the rows of the C matrix
          for (i = 0; i < M / UNROLL_L2 * UNROLL_L2; i+=UNROLL_L2){

            // We create a copy of A with the specific dimensions, based on the packed one
            if (j == 0) {
                    create_A_8xK(K, A + i, lda, &Ap[i*K]);
            }

            // Calculate the corresponding block of the C matrix (the new matrix (8x3))
            Calculate_C_Block8x3(K, &Ap[i*K], &Bp3[0], C + i + j*lda, 8, K, lda);
          }
        }

        // If not, we need to deal with the ramining rows and columns
        else {
            // Column
            process_fringes(lda, index_row, n - index_col, K, A, B + lda*index_col, C + lda*index_col);
        }

        // Row
        process_fringes(lda, M - index_row, n, K, A + index_row, B, C + index_row);
    }

    // Final case: multiple of 8 but not of 4. Explicit for enumeration).
    else if (M % 8 == 0 && n % 4 != 0){
        int index_col = n / 4 * 4;

        // Solve the fringes
        process_fringes(lda, M, n - index_col, K, A, B + lda*index_col, C + lda*index_col);
    }
}


// Compute the value of C for the remaining fringes when matrices are not multiples of 8, 4. Contains unrolling
void process_fringes(int lda, int M, int n, int K, double *A, double *B, double* C)
{
    // If one of the dimensions is null, then stop!
    if (M == 0 || n == 0) {
        return;
    }

    // Initializing indexes
    int i, j, k;

    // Compute new C values in a 2x2 approach using _m128d AVX instrinsics.
    // Initialize intrinsics data format (128d = 2 doubles)
    register __m128d c0, c1, a0, a00, b0, b00, b1, b10;

    // Main loop, i row and j colums unrolled by a factor of 2
    for(i = 0; i < M / UNROLL_L3 * UNROLL_L3; i+=UNROLL_L3){   //TESTING NEW ORDER ORIGINAL i,j
        for(j = 0; j < n / UNROLL_L3 * UNROLL_L3; j+=UNROLL_L3){

            // Load the values of C into the AVX variables (using intrinsics)
            c0 = _mm_loadu_pd(C + i + j*lda);
            c1 = _mm_loadu_pd(C + i + (j + 1)*lda);

            // Inner loop unrolled by a factor of 2
            for(k = 0; k < K; k+=2){
                // Load A and B values using AVX intrinsics
                a0 = _mm_loadu_pd(A + i + k*lda);
                b0 = _mm_load1_pd(B + k + j*lda);
                b1 = _mm_load1_pd(B + k + (j + 1)*lda);

                a00 = _mm_loadu_pd(A + i + (k+1)*lda);
                b00 = _mm_load1_pd(B + k + 1 + j*lda);
                b10 = _mm_load1_pd(B + k + 1 + (j + 1)*lda);


                // Perform the multiplication and addition at the same time using AVX2 instructions
                // From Intel: Performs a set of SIMD multiply-add computation on packed double-precision floating-point values
                c0 = _mm_fmadd_pd(a0, b0, c0);
                c1 = _mm_fmadd_pd(a0, b1, c1);
                c0 = _mm_fmadd_pd(a00, b00, c0);
                c1 = _mm_fmadd_pd(a00, b10, c1);
            }

            // Store the results in matrix C memory allocations
            _mm_storeu_pd(C + i + j*lda, c0);
            _mm_storeu_pd(C + i + (j + 1)*lda, c1);
        }

        // If we have a remaining column, process it
        if (n % 2){
            // Load the column into memory
            c0 = _mm_loadu_pd(C + i +(n - 1)*lda);

            // Perform the multiplication and addition
            for(k = 0; k < K; k++){
                // Load A and B elements into memory
                a0 = _mm_loadu_pd(A + i + k*lda);
                b0 = _mm_load1_pd(B + k + j*lda);

                // Perform the multiplication + addition via AVX2
                c0 = _mm_fmadd_pd(a0, b0, c0);
            }

            // Store the results in matrix C memory allocations
            _mm_storeu_pd(C + i + (n - 1)*lda, c0);
        }
    }

    // If we have a remaining row, process it. Unrolled loop by a factor of 2
    if (M % 2){
        // Simple double variables, no AVX can be applied (no multiple of 2)
        double c0, c1, a, b0, b1;

        // Main loop, i row and j colums unrolled by a factor of 2
        for(j = 0; j < n / UNROLL_L3 * UNROLL_L3; j+=UNROLL_L3){
            // Load C matrix values in auxiliary variables
            c0 = C[M - 1 + j*lda];
            c1 = C[M - 1 + (j+1)*lda];

            // Loop over colums
            for(k = 0; k < K; k++){
                // Load A, B, and update C
                a = A[M - 1 + k*lda];
                b0 = B[k + j*lda];
                b1 = B[k + (j+1)*lda];
                c0 += a * b0;
                c1 += a * b1;
            }

            // Store the results inside the C matrix allocation
            C[M - 1 + j*lda] = c0;
            C[M - 1 + (j+1)*lda] = c1;
        }

        // If we have remaining columns, process it.
        if (n % 2){
            // Remaining element (isolated) from the matrix
            c0 = C[M - 1 + (n - 1)*lda];

            // Inner loop over K
            for(k = 0; k < K; k++){
                a = A[M - 1 + k*lda];
                b0 = B[k +(n - 1)*lda];
                c0 += a * b0;
            }

            // Finally, store the result in the C matrix memory allocation
            C[M - 1 + (n - 1)*lda] = c0;
        }
    }
}


// Compute the value of C for Blocks of 8x4. Contains unrolling
void Calculate_C_Block8x4(int K, double *A, double* B, double *C, int lda, int ldb, int ldc)
{
    // Initialize intrinsics data format (256d = 4 doubles)
    register __m256d c0, c1, c2, c3, c4, c5, c6, c7;
    register __m256d a0, a1, b0, b1, b2, b3;

    // Load the values from C into the AVX data variables
    c0 = _mm256_loadu_pd(C);
    c1 = _mm256_loadu_pd(C + ldc);
    c2 = _mm256_loadu_pd(C + ldc*2);
    c3 = _mm256_loadu_pd(C + ldc*3);
    c4 = _mm256_loadu_pd(C + 4);
    c5 = _mm256_loadu_pd(C + 4 + ldc);
    c6 = _mm256_loadu_pd(C + 4 + ldc*2);
    c7 = _mm256_loadu_pd(C + 4 + ldc*3);

    // Main loop over k, unrolled by 4
    for(int k = 0; k < K; k++){
        // A matrix values are loaded into the AVX data type
        a0 = _mm256_loadu_pd(A);
        a1 = _mm256_loadu_pd(A + 4);
        A += lda;

        // B matrix elements are loaded and broadcasted in the same instruction
        b0 = _mm256_broadcast_sd(B);
        b1 = _mm256_broadcast_sd(B + 1);
        b2 = _mm256_broadcast_sd(B + 2);
        b3 = _mm256_broadcast_sd(B + 3);

        // Update the value of B by 4 (Unrolled)
        B += 4;

        // Perform the multiplication and addition at the same time using AVX2 instructions
        c0 = _mm256_fmadd_pd(a0, b0, c0);
        c1 = _mm256_fmadd_pd(a0, b1, c1);
        c2 = _mm256_fmadd_pd(a0, b2, c2);
        c3 = _mm256_fmadd_pd(a0, b3, c3);
        c4 = _mm256_fmadd_pd(a1, b0, c4);
        c5 = _mm256_fmadd_pd(a1, b1, c5);
        c6 = _mm256_fmadd_pd(a1, b2, c6);
        c7 = _mm256_fmadd_pd(a1, b3, c7);
    }

    // Store the results in matrix C memory allocations
    _mm256_storeu_pd(C, c0);
    _mm256_storeu_pd(C + ldc, c1);
    _mm256_storeu_pd(C + ldc*2, c2);
    _mm256_storeu_pd(C + ldc*3, c3);
    _mm256_storeu_pd(C + 4, c4);
    _mm256_storeu_pd(C + 4 + ldc, c5);
    _mm256_storeu_pd(C + 4 + ldc*2, c6);
    _mm256_storeu_pd(C + 4 + ldc*3, c7);
}


// Compute the value of C for Blocks of 8x4. Contains unrolling
void Calculate_C_Block8x8(int K, double *A, double* B, double *C, int lda, int ldb, int ldc)
{
    // Initialize intrinsics data format (256d = 4 doubles)
    register __m256d c0, c1, c2, c3, c4, c5, c6, c7;
    register __m256d a0, a1, b0, b1, b2, b3, b4, b5, b6, b7;

    // Load the values from C into the AVX data variables
    c0 = _mm256_loadu_pd(C);
    c1 = _mm256_loadu_pd(C + ldc);
    c2 = _mm256_loadu_pd(C + ldc*2);
    c3 = _mm256_loadu_pd(C + ldc*3);
    c4 = _mm256_loadu_pd(C + 4);
    c5 = _mm256_loadu_pd(C + 4 + ldc);
    c6 = _mm256_loadu_pd(C + 4 + ldc*2);
    c7 = _mm256_loadu_pd(C + 4 + ldc*3);

    // Main loop over k, unrolled by 4
    for(int k = 0; k < K; k++){
        // A matrix values are loaded into the AVX data type
        a0 = _mm256_loadu_pd(A);
        a1 = _mm256_loadu_pd(A + 4);
        A += lda;

        // B matrix elements are loaded and broadcasted in the same instruction
        b0 = _mm256_broadcast_sd(B);
        b1 = _mm256_broadcast_sd(B + 1);
        b2 = _mm256_broadcast_sd(B + 2);
        b3 = _mm256_broadcast_sd(B + 3);
        b4 = _mm256_broadcast_sd(B + 4);
        b5 = _mm256_broadcast_sd(B + 5);
        b6 = _mm256_broadcast_sd(B + 6);
        b7 = _mm256_broadcast_sd(B + 7);

        // Update the value of B by 4 (Unrolled)
        B += 8;

        // Perform the multiplication and addition at the same time using AVX2 instructions
        c0 = _mm256_fmadd_pd(a0, b0, c0);
        c1 = _mm256_fmadd_pd(a0, b1, c1);
        c2 = _mm256_fmadd_pd(a0, b2, c2);
        c3 = _mm256_fmadd_pd(a0, b3, c3);
        c4 = _mm256_fmadd_pd(a0, b4, c4);
        c5 = _mm256_fmadd_pd(a0, b5, c5);
        c6 = _mm256_fmadd_pd(a0, b6, c6);
        c7 = _mm256_fmadd_pd(a0, b7, c7);

        c0 = _mm256_fmadd_pd(a1, b0, c0);
        c1 = _mm256_fmadd_pd(a1, b1, c1);
        c2 = _mm256_fmadd_pd(a1, b2, c2);
        c3 = _mm256_fmadd_pd(a1, b3, c3);
        c4 = _mm256_fmadd_pd(a1, b4, c4);
        c5 = _mm256_fmadd_pd(a1, b5, c5);
        c6 = _mm256_fmadd_pd(a1, b6, c6);
        c7 = _mm256_fmadd_pd(a1, b7, c7);
    }

    // Store the results in matrix C memory allocations
    _mm256_storeu_pd(C, c0);
    _mm256_storeu_pd(C + ldc, c1);
    _mm256_storeu_pd(C + ldc*2, c2);
    _mm256_storeu_pd(C + ldc*3, c3);
    _mm256_storeu_pd(C + 4, c4);
    _mm256_storeu_pd(C + 4 + ldc, c5);
    _mm256_storeu_pd(C + 4 + ldc*2, c6);
    _mm256_storeu_pd(C + 4 + ldc*3, c7);
}


// Compute the value of C for Blocks of 8x3. Contains unrolling
void Calculate_C_Block8x3(int K, double *A, double* B, double *C, int lda, int ldb, int ldc)
{
    // Initialize intrinsics data format (256d = 4 doubles)
    register __m256d c0, c1, c2, c3, c4, c5;
    register __m256d a0, a1, b0, b1, b2;

    // Load the values from C into the AVX data variables
    c0 = _mm256_loadu_pd(C);
    c1 = _mm256_loadu_pd(C + ldc);
    c2 = _mm256_loadu_pd(C + ldc*2);
    c3 = _mm256_loadu_pd(C + 4);
    c4 = _mm256_loadu_pd(C + 4 + ldc);
    c5 = _mm256_loadu_pd(C + 4 + ldc*2);

    // Main loop over k, unrolled by 3
    for(int k = 0; k < K; k++){
        // A matrix values are loaded into the AVX data type
        a0 = _mm256_loadu_pd(A);
        a1 = _mm256_loadu_pd(A + 4);

        // Update A value (unrolled by lda)
        A += lda;

        // B matrix elements are loaded and broadcasted in the same instruction
        b0 = _mm256_broadcast_sd(B);
        b1 = _mm256_broadcast_sd(B + 1);
        b2 = _mm256_broadcast_sd(B + 2);

        // Update B value (unrolled by 3)
        B += 3;

        // Perform the multiplication and addition at the same time using AVX2 instructions
        c0 = _mm256_fmadd_pd(a0, b0, c0);
        c1 = _mm256_fmadd_pd(a0, b1, c1);
        c2 = _mm256_fmadd_pd(a0, b2, c2);
        c3 = _mm256_fmadd_pd(a1, b0, c3);
        c4 = _mm256_fmadd_pd(a1, b1, c4);
        c5 = _mm256_fmadd_pd(a1, b2, c5);
    }

    // Store the results in matrix C memory allocations
    _mm256_storeu_pd(C, c0);
    _mm256_storeu_pd(C + ldc, c1);
    _mm256_storeu_pd(C + ldc*2, c2);
    _mm256_storeu_pd(C + 4, c3);
    _mm256_storeu_pd(C + 4 + ldc, c4);
    _mm256_storeu_pd(C + 4 + ldc*2, c5);
}


// Compute the value of C for Blocks of 4x3. Contains unrolling
void Calculate_C_Block4x4(int K, double *A, double* B, double *C, int lda, int ldb, int ldc)
{
    // Initialize intrinsics data format (256d = 4 doubles)
    register __m256d c0, c1, c2, c3;
    register __m256d a0, b0, b1, b2, b3;

    // Load the values from C into the AVX data variables
    c0 = _mm256_loadu_pd(C);
    c1 = _mm256_loadu_pd(C + ldc);
    c2 = _mm256_loadu_pd(C + ldc*2);
    c2 = _mm256_loadu_pd(C + ldc*3);

    // Main loop over k, unrolled by 3
    for(int k = 0; k < K; k++){
        // A matrix values are loaded into the AVX data type
        a0 = _mm256_loadu_pd(A);

        // Update A value (unrolled by lda)
        A += lda;

        // B matrix elements are loaded and broadcasted in the same instruction
        b0 = _mm256_broadcast_sd(B);
        b1 = _mm256_broadcast_sd(B + 1);
        b2 = _mm256_broadcast_sd(B + 2);
        b3 = _mm256_broadcast_sd(B + 3);

        // Update B value (unrolled by 3)
        B += 4;

        // Perform the multiplication and addition at the same time using AVX2 instructions
        c0 = _mm256_fmadd_pd(a0, b0, c0);
        c1 = _mm256_fmadd_pd(a0, b1, c1);
        c2 = _mm256_fmadd_pd(a0, b2, c2);
        c3 = _mm256_fmadd_pd(a0, b3, c3);

    }

    // Store the results in matrix C memory allocations
    _mm256_storeu_pd(C, c0);
    _mm256_storeu_pd(C + ldc, c1);
    _mm256_storeu_pd(C + ldc*2, c2);
    _mm256_storeu_pd(C + ldc*3, c3);

}



/////////////////////////////////////////////
//
//                   Utils
//
/////////////////////////////////////////////


// Creates a copy with dimensions 8xk from the A matrix using pointers
void create_A_8xK(int K, double *A, int lda, double *A8k)
{
    // Main loop over k columns
    for(int j = 0; j < K; j++){
        // Initialize a pointer to the memory associated with matrix A
        double *ajlda = A + j*lda;

        // Unrolled loop for 8: creates a copy into A8k
        *A8k = *ajlda;
        *(A8k+1) = *(ajlda+1);
        *(A8k+2) = *(ajlda+2);
        *(A8k+3) = *(ajlda+3);
        *(A8k+4) = *(ajlda+4);
        *(A8k+5) = *(ajlda+5);
        *(A8k+6) = *(ajlda+6);
        *(A8k+7) = *(ajlda+7);

        // Update the value for next iteration of j
        A8k+= 8;
    }
}


// Creates a copy with dimensions 4xk from the A matrix using pointers
void create_A_4xK(int K, double *A, int lda, double *A4k)
{
    // Main loop over k columns
    for(int j = 0; j < K; j++){
        // Initialize a pointer to the memory associated with matrix A
        double *ajlda = A + j*lda;

        // Unrolled loop for 8: creates a copy into A8k
        *A4k = *ajlda;
        *(A4k+1) = *(ajlda+1);
        *(A4k+2) = *(ajlda+2);
        *(A4k+3) = *(ajlda+3);

        // Update the value for next iteration of j
        A4k+= 4;
    }
}

// Creates a copy of B such that it is packed into a row-major format
void row_major_B(int K, double *B, int ldb, double *BT4)
{
    // Initialize all the pointers for packing the B matrix
    double *b0 = B, *b1 = B + ldb, *b2 = B + 2*ldb, *b3 = B + 3*ldb;

    // Main loop for packing B
    for(int i = 0; i < K; i++){
        *BT4++ = *b0++;
        *BT4++ = *b1++;
        *BT4++ = *b2++;
        *BT4++ = *b3++;
    }
}

// Creates a copy of B such that it is packed into a row-major format
// (different dimension of the original row_major_B function)
void row_major_Bx3(int K, double *B, int ldb, double *BT3)
{
    // Initialize all the pointers for packing the B matrix
    double *b0 = B, *b1 = B + ldb, *b2 = B + 2*ldb;

    // Main loop for packing B
    for(int i = 0; i < K; i++){
        *BT3++ = *b0++;
        *BT3++ = *b1++;
        *BT3++ = *b2++;
    }
}






