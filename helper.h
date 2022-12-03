#ifndef HELPER
#define HELPER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#if 0
typedef int           int32_t;
typedef long long  int64_t;
#endif

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;


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
    for (int i = 0; i < size; i++)
        data[i] = rand() / (T)RAND_MAX;
}

/**
 * Initialize the `data` array, which has `size` elements:
 * frac% of them are NaNs and (1-frac)% are random values.
 * 
 */
void randomMask(char* data, int size, float frac) {
    for (int i = 0; i < size; i++) {
        float r = rand() / (float)RAND_MAX;
        data[i] = (r >= frac) ? 1 : 0;
    }
}

// error for matmul: 0.02
template<class T>
bool validate(T* A, T* B, unsigned int sizeAB, const T ERR){
    for(int i = 0; i < sizeAB; i++) {
        T curr_err = fabs( (A[i] - B[i]) / max(A[i], B[i]) ); 
        if (curr_err >= ERR) {
            printf("INVALID RESULT at flat index %d: %f vs %f\n", i, A[i], B[i]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}

#endif
