#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 
#include <cstdlib>

#include <curand.h>
#include <cublas_v2.h>

// nvcc main-cublas.cu -lcublas -lcurand

#define GPU_RUNS 25

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

template<typename T>
void GPU_fill_rand(T *A, int nr_rows_A, int nr_cols_A) {
    return;
}


// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
template<>
void GPU_fill_rand<float>(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

template<>
void GPU_fill_rand<double>(double *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniformDouble(prng, A, nr_rows_A * nr_cols_A);
}


void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;

     // Do the actual multiplication
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template<class T> 
void callCublas() {
    return;
}

template<class T> 
void gpu_blas_mmul(cublasHandle_t &handle, const T *A, const T *B, T *C, const int m, const int k, const int n) {
    return;
}

template<>
void gpu_blas_mmul<float>(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;

     // Do the actual multiplication
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void gpu_blas_mmul<double>(cublasHandle_t &handle, const double *A, const double *B, double *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;
     const double alf = 1;
     const double bet = 0;
     const double *alpha = &alf;
     const double *beta = &bet;

     // Do the actual multiplication
     cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
 
// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
template<class T>
void gpu_blas_mmul_rep(const T *A, const T *B, T *C, const int m, const int k, const int n) {
#if 0
     int lda=m,ldb=k,ldc=m;
     const T alf = 1;
     const T bet = 0;
     const T *alpha = &alf;
     const T *beta = &bet;
#endif
     // Create a handle for CUBLAS
     cublasHandle_t handle;
     cublasCreate(&handle);
 
      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
     
     // Do the actual multiplication
      for(int i=0; i < GPU_RUNS; i++) {
            gpu_blas_mmul<T>(handle, A, B, C, m, k, n);

//            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      }
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

      float microsecPerMatrixMul = elapsed; 
      double flopsPerMatrixMul = 2.0 * m * k * n;
      double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

      printf("CUBLAS runs in: %lu microsecs, GFlops/sec: %f\n", elapsed, gigaFlops);

     // Destroy the handle
     cublasDestroy(handle);
}

template<class T>
void runMMM(const int HEIGHT_A, const int WIDTH_A, const int WIDTH_B) {
     // Allocate 3 arrays on CPU
     int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
 
     // for simplicity we are going to use square arrays
     nr_rows_A = HEIGHT_A;
     nr_cols_A = WIDTH_A;
     nr_rows_B = nr_cols_A;
     nr_cols_B = WIDTH_B;
     nr_rows_C = nr_rows_A;
     nr_cols_C = nr_cols_B;
 
     T *h_A = (T *)malloc(nr_rows_A * nr_cols_A * sizeof(T));
     T *h_B = (T *)malloc(nr_rows_B * nr_cols_B * sizeof(T));
     T *h_C = (T *)malloc(nr_rows_C * nr_cols_C * sizeof(T));
 
     // Allocate 3 arrays on GPU
     T *d_A, *d_B, *d_C;
     cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(T));
     cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(T));
     cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(T));
 
     // Fill the arrays A and B on GPU with random numbers
     GPU_fill_rand<T>(d_A, nr_rows_A, nr_cols_A);
     GPU_fill_rand<T>(d_B, nr_rows_B, nr_cols_B);
 
     // Optionally we can copy the data back on CPU and print the arrays
     cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(T),cudaMemcpyDeviceToHost);
     cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(T),cudaMemcpyDeviceToHost);
 
     // Multiply A and B on GPU
     gpu_blas_mmul_rep<T>(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
 
     // Copy (and print) the result on host memory
     cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(T),cudaMemcpyDeviceToHost);
 
     //Free GPU memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
 
     // Free CPU memory
     free(h_A);
     free(h_B);
     free(h_C);
 
}

int main (int argc, char * argv[]) {
    if (argc != 4) {
        printf("Usage: %s heiht-A width-A width-B\n", argv[0]);
        exit(1);
    }
    const int HEIGHT_A = atoi(argv[1]);
    const int WIDTH_A  = atoi(argv[2]);
    const int WIDTH_B  = atoi(argv[3]);

     runMMM< float>(HEIGHT_A, WIDTH_A, WIDTH_B);
     runMMM<double>(HEIGHT_A, WIDTH_A, WIDTH_B);
 
     return 0;
}
