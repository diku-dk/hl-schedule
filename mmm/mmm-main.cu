#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#include "mmm-kernels.cu.h"

using namespace std;

#define GPU_RUNS    100

//    #define TILE     16//16
//    #define Ty  16
//    #define Tx  16
//    #define Ry  4
//    #define Rx  4
//    #define Tk  16
//    #define Rk  16 //32


/////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

template<class T>
void randomInit(T* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = (T) ( rand() / (float)RAND_MAX );
}


template<class T>
void matMult(T* A, T* B, T* C, int colsA, int rowsA, int colsB) {
  for(int i = 0; i < rowsA; i++) {
    for(int j = 0; j < colsB; j++) {
      float sum = 0.0;
      for(int k = 0; k < colsA; k++) {
        sum += A[i*colsA + k] * B[k * colsB + j];
      }
      C[i * colsB + j] = sum;
    }
  } 
}

template<class T>
bool validate(T* A, T* B, unsigned int sizeAB){
    for(int i = 0; i < sizeAB; i++)
      if (fabs(A[i] - B[i]) > 0.02) { //0.0007){
        printf("INVALID RESULT %d %f %f\n", i, A[i], B[i]);
        return false;
      }
    printf("VALID RESULT!\n");
    return true;
}


// naive kernel, i.e., the only tiling performed is on the grid;
//   no shared or private memory is used.
template< typename T, int TL>
__host__ void runNaive(  int height_A, int width_A, int width_B,
                T* d_A, T* d_B, T* d_C, T* h_C
             ) {

    unsigned long long mem_size_C = height_A*width_B*sizeof(T);
    cudaMemset (d_C, 0, mem_size_C );

    // setup execution parameters
    int  dimy = (height_A + TL - 1) / TL; 
    int  dimx = (width_B  + TL - 1) / TL;

    dim3 block(TL, TL, 1);
    dim3 grid (dimx, dimy, 1);

    // dry run
    mmmNaiveKer<T> <<< grid, block >>>(d_A, d_B, d_C, height_A, width_B, width_A);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    
    for(int i=0; i<GPU_RUNS; i++) {
        mmmNaiveKer<T> <<< grid, block >>>(d_A, d_B, d_C, height_A, width_B, width_A);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    float microsecPerMatrixMul = elapsed; 
    double flopsPerMatrixMul = 2.0 * height_A * width_B * width_A; 
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

    printf("GPU Naive MMM version runs in: %lu microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);

    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);    
}

//  register tile on A, double block tile on B
template<class T, int TL>
void runAsymetricBlkRegTile( 
        int HEIGHT_A, int WIDTH_A, int WIDTH_B,
        T* d_A, T* d_B, T* d_C, T* ref_C
    ) {

    unsigned long long size_C = HEIGHT_A * WIDTH_B;
    unsigned long long mem_size_C = size_C * sizeof(T);
    T* h_C = (T*) malloc(mem_size_C);

    // setup execution parameters
    int  dimy = ceil( ((float)HEIGHT_A)/TL ); 
    int  dimx = ceil( ((float) WIDTH_B)/(TL*TL) );
    dim3 block(TL, TL, 1);
    dim3 grid (dimx, dimy, 1);

    { // one dry run
        mmmAsymBlkRegKer<T,TL> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }
    
    cudaMemset(d_C, 0, mem_size_C);
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
      
    for(int i=0; i<GPU_RUNS; i++) {
        mmmAsymBlkRegKer<T,TL> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    float microsecPerMatrixMul = elapsed; 
    double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

    printf("GPU Asymetric Blk-Reg Tiled MMM version runs in: %lu microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);

    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
       
    validate<T>(ref_C, h_C, size_C);
    free(h_C);
}

//  symmetric block-and-register tiling with inner dimension sequential
template<class T, int Ty, int Ry, int Tx, int Rx, int Tk>
void runSymetricBlkRegTileInnSeq( 
        int HEIGHT_A, int WIDTH_A, int WIDTH_B,
        T* d_A, T* d_B, T* d_C, T* ref_C
    ) {

    unsigned long long size_C = HEIGHT_A * WIDTH_B;
    unsigned long long mem_size_C = size_C * sizeof(T);
    T* h_C = (T*) malloc(mem_size_C);

    // setup execution parameters
    int  dimy = ceil( ((float)HEIGHT_A)/(Ty*Ry) ); 
    int  dimx = ceil( ((float) WIDTH_B)/(Tx*Rx) );
    dim3 block(Tx, Ty, 1);
    dim3 grid (dimx, dimy, 1);

    { // one dry run
        mmmSymBlkRegInnSeqKer<T,Ty,Ry,Tx,Rx,Tk> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }

    cudaMemset(d_C, 0, mem_size_C);

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
      
    for(int i=0; i<GPU_RUNS; i++) {
        mmmSymBlkRegInnSeqKer<T,Ty,Ry,Tx,Rx,Tk> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

    float microsecPerMatrixMul = elapsed; 
    double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

    printf( "GPU Symetric Blk-Reg Tiled MMM version with seq Inner runs in: %lu microsecs, GFlops/sec: %.2f\n"
          , elapsed, gigaFlops );


    // copy result from device to host and validate
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    validate<T>(ref_C, h_C, size_C);
    free(h_C);
}


//  symmetric block-and-register tiling with inner dim parallel
template<class T, int Ty, int Ry, int Tx, int Rx, int Tk, int Rk>
void runSymetricBlkRegTileAllPar( 
        int HEIGHT_A, int WIDTH_A, int WIDTH_B,
        T* d_A, T* d_B, T* d_C, T* ref_C
    ) {

    unsigned long long size_C = HEIGHT_A * WIDTH_B;
    unsigned long long mem_size_C = size_C * sizeof(T);
    T* h_C = (T*) malloc(mem_size_C);

    // setup execution parameters
    // (Ty,Ry,Tx,Rx,Tk) are as before
    // Rk is not a register tile size per say, it just means that
    //    a factor Rk*Tk of the WIDTH_A dimension is going to be
    //    sequentialized, and (WIDTH_A / (Rk*Tk)) is going to be
    //    parallelized
    int  dimy = ceil( ((float)HEIGHT_A)/(Ty*Ry) ); 
    int  dimx = ceil( ((float) WIDTH_B)/(Tx*Rx) );
    int  dimz = ceil( ((float) WIDTH_A)/(Tk*Rk) );
    dim3 block(Tx, Ty, 1);
    dim3 grid (dimx, dimy, dimz);

    { // one dry run
        mmmSymBlkRegAllParKer<T,Ty,Ry,Tx,Rx,Tk,Rk> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }

    cudaMemset(d_C, 0, mem_size_C);

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
      
    for(int i=0; i<GPU_RUNS; i++) {
        cudaMemset(d_C, 0, mem_size_C);
        mmmSymBlkRegAllParKer<T,Ty,Ry,Tx,Rx,Tk,Rk> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

    float microsecPerMatrixMul = elapsed; 
    double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

    printf( "GPU Symetric Blk-Reg Tiled MMM version with all parallel runs in: %lu microsecs, GFlops/sec: %.2f\n"
          , elapsed, gigaFlops );


    // copy result from device to host and validate
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    validate<T>(ref_C, h_C, size_C);
    free(h_C);
}

template<class T, int TL, int REG>
void runAll ( int height_A, int width_A, int width_B ) {

    srand(2006);
 
    // 1. allocate host memory for the two matrices
    unsigned long long size_A = width_A * height_A;
    unsigned long long mem_size_A = sizeof(T) * size_A;
    T* h_A = (T*) malloc(mem_size_A);
 
    unsigned long long size_B = width_B * width_A;
    unsigned long long mem_size_B = sizeof(T) * size_B;
    T* h_B = (T*) malloc(mem_size_B);
 
    // 2. initialize host memory
    randomInit<T>(h_A, size_A);
    randomInit<T>(h_B, size_B);
    
    // 3. allocate device memory
    T* d_A;
    T* d_B;
    cudaMalloc((void**) &d_A, mem_size_A);
    cudaMalloc((void**) &d_B, mem_size_B);
 
    // 4. copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
 
    // 5. allocate host memory for the result C
    unsigned int size_C = height_A * width_B;
    unsigned int mem_size_C = sizeof(T) * size_C;
    //T* h_C   = (T*) malloc(mem_size_C);
    T* ref_C = (T*) malloc(mem_size_C);
 
    // 6. allocate device memory for the result
    T *d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    printf("Sizes are: (HeightA, WidthB, WidthA)=(%d, %d, %d)\n", height_A, width_B, width_A);

    runNaive<T, TL>( height_A, width_A, width_B, d_A, d_B, d_C, ref_C );

    // with TL = 24 => error. why?
    runAsymetricBlkRegTile<T, TL>( height_A, width_A, width_B, d_A, d_B, d_C, ref_C );

    runSymetricBlkRegTileInnSeq<T, TL, REG, TL, REG, TL>( height_A, width_A, width_B, d_A, d_B, d_C, ref_C );

    runSymetricBlkRegTileAllPar<T, TL, REG, TL, REG, TL, TL>( height_A, width_A, width_B, d_A, d_B, d_C, ref_C );

    free(h_A);
    free(h_B);
    free(ref_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main (int argc, char * argv[]) {
    if (argc != 4) {
        printf("Usage: %s heiht-A width-A width-B\n", argv[0]);
        exit(1);
    }
    const int HEIGHT_A = atoi(argv[1]);
    const int WIDTH_A  = atoi(argv[2]);
    const int WIDTH_B  = atoi(argv[3]);

    runAll<float, 16, 4> ( HEIGHT_A, WIDTH_A, WIDTH_B );
    runAll<double,16, 4> ( HEIGHT_A, WIDTH_A, WIDTH_B );
}
