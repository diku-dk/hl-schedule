#ifndef BMMM_KERNELS
#define BMMM_KERNELS

#define SPEC_NAN 33333333.3333

/**
 * We use ElTp for the (generic) array-element type,
 * and T for the generic tile size.
 */

template <class ElTp> __global__ 
void bmmmNaiveKer ( ElTp* A, ElTp* B
                  , ElTp* X, ElTp* Y
                  , const int M,  const int K1
                  , const int K2, const int N
) {
  const int i  = blockIdx.z * blockDim.z + threadIdx.z;
  const int j1 = blockIdx.y * blockDim.y + threadIdx.y;
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;

  // check bounds
  if( (i >= M) || (j1 >= K1) || (j2 >= K2) )
    return;

  ElTp acc = 0.0f;

  for(int q=0; q<N; q++) { // reduction
    if( X[i*N + q] != SPEC_NAN) {
      ElTp a = A[j1*N + q];
      ElTp b = B[q*K2 + j2];
      acc += a*b;
    }
  }
  Y[i*K1*K2 + j1*K2 + j2] = acc;
}

/**
 * Array dimensions are as in goldenSeq;
 * X_t is the transposed on X: [N][M]ElTp.
 * Assumes:
 *    (1) blockDim.z == Z
 *    (2) blockDim.y * blockDim.x >= R
 */
template <class ElTp, int Z, int T, int R> __global__
void bmmmTiledKer ( ElTp* A,      ElTp* B
                  , ElTp* X_tr,   ElTp* Y
                  , const int M,  const int K1
                  , const int K2, const int N
) {
  extern __shared__ char sh_mem[];
  volatile ElTp* Xsh_tr = (ElTp*)sh_mem;
  //__shared__ ElTp Xsh_tr[Z][R];
  ElTp acc[R];

  const int ii  = blockIdx.z * blockDim.z;
  const int jj1 = blockIdx.y * blockDim.y;
  const int jj2 = blockIdx.x * blockDim.x;
  const int j1  = jj1 + threadIdx.y;
  const int j2  = jj2 + threadIdx.x;
  const int i   = (ii  + threadIdx.z) * R;
  const int flat_thid = threadIdx.y * blockDim.x + threadIdx.x;

  #pragma unroll
  for(int s=0; s<R; s++)
    acc[s] = 0;

  for(int q=0; q<N; q++) {

    // read A and B to registers.
    ElTp a = 0, b = 0;
    if(j1 < K1)
      a = A[j1*N + q];
    if(j2 < K2)
      b = B[q*K2 + j2];

    ElTp ab = a*b;

    // collectively read R elements from X_tr
    ElTp x = SPEC_NAN;
    
    if(i < M && flat_thid < R) {
      x = X_tr[q*M + i + flat_thid];
    }

    if(flat_thid < R) {
      //Xsh_tr[threadIdx.z][flat_thid] = x;
      Xsh_tr[threadIdx.z*R + flat_thid] = x;
    }
    __syncthreads();

    #pragma unroll
    for(int s=0; s<R; s++) {
#if 0
      ElTp v = 1.0 - (Xsh_tr[threadIdx.z][s] == SPEC_NAN);
      acc[s] += ab * v;
#else
      //if(Xsh_tr[threadIdx.z][s] != SPEC_NAN)
      if(Xsh_tr[threadIdx.z*R + s] != SPEC_NAN)
        acc[s] += ab;
#endif
    }
    __syncthreads();
  }

  #pragma unroll
  for(int s=0; s<R; s++) {
    const int ips = i + s;
    if(ips < M && j1 < K1 && j2 < K2)
      Y[ips*K1*K2 + j1*K2 + j2] = acc[s];
  }
}


/**
 * Array dimensions are as in goldenSeq;
 * X_t is the transposed on X: [N][M]ElTp.
 * Assumes:
 *    (1) blockDim.y * blockDim.x >= R
 *    (2) blockDim.x == blockDim.y == T
 */
template <class ElTp, int T, int R> __global__
void bmmmTiledSKer( ElTp* A,      ElTp* B
                  , ElTp* X_tr,   ElTp* Y
                  , const int M,  const int K1
                  , const int K2, const int N
) {
  extern __shared__ char sh_mem[];
  volatile ElTp* Xsh_tr = (ElTp*)sh_mem;
  //__shared__ ElTp Xsh_tr[Z][R];
  ElTp acc[R];

  const int ii  = blockIdx.z;
  const int jj1 = blockIdx.y * T;
  const int jj2 = blockIdx.x * T;
  const int j1  = jj1 + threadIdx.y;
  const int j2  = jj2 + threadIdx.x;
  const int i   = ii * R;
  const int flat_thid = threadIdx.y * T + threadIdx.x;

  #pragma unroll
  for(int s=0; s<R; s++)
    acc[s] = 0;

  for(int q=0; q<N; q++) {

    // collectively read R elements from X_tr
    ElTp x = SPEC_NAN;
    
    if(i < M && flat_thid < R) {
      x = X_tr[q*M + i + flat_thid];
    }

    if(flat_thid < R) {
      //Xsh_tr[threadIdx.z][flat_thid] = x;
      Xsh_tr[flat_thid] = x;
    }

    // read A and B to registers.
    ElTp a = 0, b = 0;
    if(j1 < K1)
      a = A[j1*N + q];
    if(j2 < K2)
      b = B[q*K2 + j2];

    ElTp ab = a*b;

    __syncthreads();

    #pragma unroll
    for(int s=0; s<R; s++) {
#if 0
      ElTp v = 1.0 - (Xsh_tr[s] == SPEC_NAN);
      acc[s] += ab * v;
#else
      //if(Xsh_tr[threadIdx.z][s] != SPEC_NAN)
      if(Xsh_tr[s] != SPEC_NAN)
        acc[s] += ab;
#endif
    }
    __syncthreads();
  }

  #pragma unroll
  for(int s=0; s<R; s++) {
    const int ips = i + s;
    if(ips < M && j1 < K1 && j2 < K2)
      Y[ips*K1*K2 + j1*K2 + j2] = acc[s];
  }
}

/**
 * This is not the optimal version as it
 *   does not efficiently sequentializes.
 * Assumes:
 *    blockDim.y == blockDim.x == T
 */
template <class ElTp, int T> 
__global__ void matTransposeTiledKer(ElTp* A, ElTp* A_tr, const int heightA, const int widthA) {
  __shared__ ElTp tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x; 
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      A_tr[y*heightA + x] = tile[threadIdx.x][threadIdx.y];
}

#endif
