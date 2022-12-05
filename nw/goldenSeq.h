#ifndef GOLDEN
#define GOLDEN

/**
 * Denote: N = n*b+1
 * Input: 
 *   R    : [N][N]T
 *   Arow : [N]T
 *   Acol : [N-1]T
 * Result:
 *   A : [N][N]T
 */
template<class T, int B>
T* goldenSeq(T* Arow, T* Acol, T* R
              , const int q, const int b
) {
  const int N = q*b + 1;
  T* A = (T*)malloc(N*N*sizeof(T));

  // place Arow and Acol as the first row and column of A
  for(int i=0; i<N; i++) { A[i] = Arow[i]; }
  for(int i=1; i<N; i++) { A[i*N] = Acol[i]; }

  // target kernel
  for(int i=1; i<N; i++) {
    for(int j=1; j<N; j++) {
        // A[i,j] = f(A[i-1,j], A[i, j-1], A[i-1,j-1], R[i,j])
        A[i*N+j] =  A[(i-1)*N + j] + A[i*N + j-1] +
                    A[(i-1)*N + j-1] + R[i*N + j];
    }
  }

  return A;
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
 * W(A) = A[1:,1:].split(1).split(0).antidiag(i).mapPar(0 -> G1.x)
 *    // Produces LMAD: i*b+N+1 + { (i+1 : n*b−b), (b : n), (b : 1) }
 * // Don't know how to represent the horizontal and vertical columns:
 * // R_hor(A) = SeqU_{i=0..q-1} ( i*b + {(i+1 : n*b−b), (1 : n), (b+1 : 1)} )
 * // R_ver(A) = SeqU_{i=0..q-1} ( i*b + {(i+1 : n*b−b), (b+1 : n), (1 : 1)} )
 *
 * @Block B1(tid.x < b; fv: i=0..q-1):
 * Ash = new shmem(b+1, b+1); // [b+1][b+1]
 * Ash[0,:] = glb2sh(R_hor(A)[@])
 * Ash[:,0] = glb2sh(R_ver(A)[@])
 * W(Ash) = SeqU_{j=0..b-1}( Ash[1:,1:].antidiag(j).mapPar(0 -> B1.x) ) U
 *          SeqU_{j=b-1..0}( ...mapPar(0 -> B1.x) )
 * update( W(A).g[@], Ash[1:,1:] )
 *
 * @Grid G2(...)
 * ...
 */

#endif
