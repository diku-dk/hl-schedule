#ifndef GOLDEN
#define GOLDEN

/**
 * Denote: N = q*b+1
 * Input: 
 *   R    : [N][N]T
 *   X    : [N][N]T
 * In-Place Result:
 *   X    : [N][N]T
 */
void goldenSeq( int* X, int* R, const int N, const int penalty ) {
    //#hlsched@GPU nwCudaSched(N, X)
    for(uint64_t i=1; i<N; i++) {
        for(uint64_t j=1; j<N; j++) {
            X[i*N+j] = max3( X[ (i-1)*N + (j-1) ] + R[ i*N + j ]
                           , X[ i*N + (j-1) ] - penalty
                           , X[ (i-1)*N + j ] - penalty
                           );
        }
    }
}

#endif
