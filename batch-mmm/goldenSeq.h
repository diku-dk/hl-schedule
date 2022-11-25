#include <math.h>

/**
 * This is a special kind of batch matrix multiplication, in
 * which the same two matrices are multiplied, but the values
 * of the two matrices are filtered under a mask that differs
 * across the batch.
 * Input arrays with dimensions:
 *    A : [K1][N]T
 *    B : [N][K2]T
 *    X :  [M][N]T  (the mask)
 * Result:
 *    Y : [M][K1][K2]T
 * T is some numeric type (single/double precision floats).
 * 
 * Temporal locality:
 *    the index of each array read is invariant to two parallel dimensions.
 **/
template<class T>
void goldenSeq(T* A, T* B, T* X, T* Y, const int M, const int K1, const int K2, const int N) {
    for(int i=0; i<M; i++) { // parallel
        for(int j1=0; j1<K1; j1++) { // parallel
            for(int j2=0; j2<K2; j2++) { // parallel
                float acc = 0.0;
                for(int q=0; q<N; q++) { // reduction
                    float a = A[j1*N + q];
                    float b = B[q*K2 + j2];
                    if( ! isnan( X[i*N + q] ) ) {
                        acc += a*b;
                    }
                }
                Y[i*K1*K2 + j1*K2 + j2] = acc;
            }
        }
    }
}

