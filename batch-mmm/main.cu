#include "../helper.h"
#include "kernels.cu.h"
#include "goldenSeq.h"

using namespace std;

#define GPU_RUNS    100
#define ERR         0.00012

/**
 * Naive kernel: the only tiling performed is on the grid;
 *               no shared or private memory is used.
 * d_A, d_B, d_X are the input matrices stored on the device;
 * ref_Y is the reference-result matrix computed with goldenSeq
 *       and stored on the host.
 * d_Y is the result array, which is to be computed on the
 *     device and validated against ref_Y.
 * For matrix sizes, please see goldenSeq.
 * T is the generic (numeric) array-element type.
 * TL is the block size on the X and Y dimensions and
 * TZ is the block size on the Z dimension.
 */
template< typename T, int TZ, int TL> __host__
void runNaive ( T* d_A, T* d_B, T* d_X
              , T* ref_Y, T* d_Y
              , const int M, const int K1
              , const int K2, const int N
) {
    unsigned long long size_Y = M*K1*K2;
    unsigned long long mem_size_Y = size_Y * sizeof(T);
    cudaMemset (d_Y, 0, mem_size_Y );

    // setup execution parameters
    const int  dimz = (M  + TZ - 1) / TZ;
    const int  dimy = (K1 + TL - 1) / TL; 
    const int  dimx = (K2 + TL - 1) / TL;

    dim3 block(TL, TL, TZ);
    dim3 grid (dimx, dimy, dimz);

    // dry run
    bmmmNaiveKer<T> <<< grid, block >>>(d_A, d_B, d_X, d_Y, M, K1, K2, N);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    
    for(int i=0; i<GPU_RUNS; i++) {
        bmmmNaiveKer<T> <<< grid, block >>>(d_A, d_B, d_X, d_Y, M, K1, K2, N);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    float microsecPerMatrixMul = elapsed; 
    double flopsPerMatrixMul = 3.0 * M * K1 * K2 * N; // 3.0 or 4.0 ?
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

    printf("GPU Naive BMMM version runs in: %lu microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);

    T* h_Y = (T*) malloc(mem_size_Y);
    cudaMemcpy(h_Y, d_Y, mem_size_Y, cudaMemcpyDeviceToHost);
    validate<T>(ref_Y, h_Y, size_Y, ERR);
    free(h_Y);
}

template<typename T, int TILE>
void runTranspose(T* d_A, T* d_A_tr, const int heightA, const int widthA) {
    const int dimy = (heightA + TILE - 1) / TILE;
    const int dimx = (widthA  + TILE - 1) / TILE;
    dim3 block(TILE, TILE, 1);
    dim3 grid (dimx, dimy, 1);
    matTransposeTiledKer<T, 32><<<grid, block>>>(d_A, d_A_tr, heightA, widthA);
}

template< typename T, int TZ, int TL, int TR> __host__
void runTiled ( T* d_A, T* d_B
              , T* d_X, T* d_X_tr
              , T* ref_Y, T* d_Y
              , const int M, const int K1
              , const int K2, const int N
) {
    unsigned long long size_Y = M*K1*K2;
    unsigned long long mem_size_Y = size_Y * sizeof(T);
    cudaMemset (d_Y, 0, mem_size_Y );

    // setup execution parameters
    const int  dimz = (M  + TZ*TR - 1) / (TZ*TR);
    const int  dimy = (K1 + TL - 1) / TL;
    const int  dimx = (K2 + TL - 1) / TL;

    dim3 block(TL, TL, TZ);
    dim3 grid (dimx, dimy, dimz);

    // dry run
    runTranspose<T,32>(d_X, d_X_tr, M, N);
    bmmmTiledKer<T, TZ, TL, TR><<< grid, block >>>(d_A, d_B, d_X_tr, d_Y, M, K1, K2, N);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    
    for(int i=0; i<GPU_RUNS; i++) {
        runTranspose<T, 32>(d_X, d_X_tr, M, N);
        bmmmTiledKer<T, TZ, TL, TR><<< grid, block >>>(d_A, d_B, d_X_tr, d_Y, M, K1, K2, N);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    float  microsecPerMatrixMul = elapsed; 
    double flopsPerMatrixMul = 3.0 * M * K1 * K2 * N; // 3.0 or 4.0 ?
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

    printf("GPU RegTiled BMMM version runs in: %lu microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);

    T* h_Y = (T*) malloc(mem_size_Y);
    cudaMemcpy(h_Y, d_Y, mem_size_Y, cudaMemcpyDeviceToHost);
    validate<T>(ref_Y, h_Y, size_Y, ERR);
    free(h_Y);
}

/**
 * This will run all code versions
 * (and summarize the performance in GFlops). 
 */
template<class T, int TZ, int TL, int TR>
void runAll ( const int M, const int K1
            , const int K2, const int N
) {

    srand(2022);
 
    // 1. allocate host memory for the four matrices A, B, X, Y
    unsigned long long size_A = K1 * N;
    unsigned long long mem_size_A = sizeof(T) * size_A;
    T* h_A = (T*) malloc(mem_size_A);
 
    unsigned long long size_B = N * K2;
    unsigned long long mem_size_B = sizeof(T) * size_B;
    T* h_B = (T*) malloc(mem_size_B);

    unsigned long long size_X = M * N;
    unsigned long long mem_size_X = sizeof(T) * size_X;
    T* h_X = (T*) malloc(mem_size_X);

    unsigned long long size_Y = M * K1 * K2;
    unsigned long long mem_size_Y = sizeof(T) * size_Y;
    T* h_Y = (T*) malloc(mem_size_Y);
 
    // 2. initialize input arrays in host memory
    randomInit<T>(h_A, size_A);
    randomInit<T>(h_B, size_B);
    randomInitWithNaNs<T>(h_X, SPEC_NAN, size_X, 0.1);
    
    // 3. allocate device memory
    T *d_A, *d_B, *d_X, *d_X_tr, *d_Y;
    cudaMalloc((void**) &d_A,    mem_size_A);
    cudaMalloc((void**) &d_B,    mem_size_B);
    cudaMalloc((void**) &d_X,    mem_size_X);
    cudaMalloc((void**) &d_X_tr, mem_size_X);
    cudaMalloc((void**) &d_Y,    mem_size_Y);
 
    // 4. copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, mem_size_X, cudaMemcpyHostToDevice);
  
    printf("Sizes are: (M, K1, K2, N)=(%d, %d, %d, %d)\n", M, K1, K2, N);

    // 5. compute golden sequential
    goldenSeq<T>(h_A, h_B, h_X, h_Y, M, K1, K2, N);

    // 6. compute the naive GPU version
    runNaive<T, TZ, TL>(d_A, d_B, d_X, h_Y, d_Y, M, K1, K2, N);

    // 7. compute the register-tiled version
    runTiled<T, TZ, TL, TR>(d_A, d_B, d_X, d_X_tr, h_Y, d_Y, M, K1, K2, N);

    free(h_A);
    free(h_B);
    free(h_X);
    free(h_Y);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_X);
    cudaFree(d_X_tr);
    cudaFree(d_Y);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main (int argc, char * argv[]) {
    if (argc != 5) {
        printf("Usage: %s M K1 K2 N\n", argv[0]);
        exit(1);
    }
    const int M  = atoi(argv[1]);
    const int K1 = atoi(argv[2]);
    const int K2 = atoi(argv[3]);
    const int N  = atoi(argv[4]);

    runAll<float, 2, 8, 30> ( M, K1, K2, N );
    runAll<double,2, 8, 30> ( M, K1, K2, N );
}
