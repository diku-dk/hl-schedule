#ifndef GOLDEN
#define GOLDEN

/**
 * Denote: N = q*b+1
 * Input: 
 *   R    : [N-1][N-1]T
 *   X    : [N][N]T
 * In-Place Result
 *   X    : [N][N]T
 */
void goldenSeq( int* X, int* R, const int N, const int penalty ) {
    for(uint64_t i=1; i<N; i++) {
        for(uint64_t j=1; j<N; j++) {
            X[i*N+j] = max3( X[ (i-1)*N + (j-1) ] + R[ i*N + j ]
                           , X[ i*N + (j-1) ] - penalty
                           , X[ (i-1)*N + j ] - penalty
                           );
        }
    }
}


/**
 * An LMAD L = off + { (n_1 : s_1), ... (n_k, s_k) }
 *   denotes the set of 1D points (memory locations):
 *   { off + i_1*s_1 + ... + i_k*s_k |
 *     0 <= i_1 < n_1, ..., 0 <= i_k < n_k }
 */

/**
 * // HL-Schedule of kernel "not-implemented-yet" (in kernels.cu.h)
 * // The grid is called inside a seq loop SeqU_{i=0..q-1} { ... }
 * //   how to represent that?
 *
 * @Grid G1(bid.x < i+1; fv: i=0..q-1):
 * assumes N = q*b + 1
 *
 * W(X) = X[1:,1:].split(1).split(0).reorder([0,2,1,3]).antidiag(i).mapPar(0 -> G1.x)
 *    // Produces LMAD: i*b+N+1 + { (i+1 : n*b−b), (b : n), (b : 1) }
 * // Don't know how to represent the horizontal and vertical columns:
 * // R_hor(X) = SeqU_{i=0..q-1} ( i*b + {(i+1 : n*b−b), (1 : n), (b+1 : 1)} )
 * // R_ver(X) = SeqU_{i=0..q-1} ( i*b + {(i+1 : n*b−b), (b+1 : n), (1 : 1)} )
 *
 * @Block B1(tid.x < b; fv: i=0..q-1):
 * Xsh = new shmem(b+1, b+1); // [b+1][b+1]
 * Xsh[0,:] = glb2sh(R_hor(X)[@])
 * Xsh[:,0] = glb2sh(R_ver(X)[@])
 * W(Xsh) = SeqU_{j=0..b-1}( Xsh[1:,1:].antidiag(j).mapPar(0 -> B1.x) ) U
 *          SeqU_{j=b-1..0}( ...mapPar(0 -> B1.x) )
 * update( W(X).g[@], Xsh[1:,1:] )
 *
 * @Grid G2(...)
 * ...
 */

#endif
