#ifndef BMMM_KERNELS
#define BMMM_KERNELS

/**
 * We use ElTp for the (generic) array-element type,
 * and T for the generic tile size.
 */

template <class ElTp> __global__ 
void spmvNaiveKer ( uint64_t*  rows
                  , uint32_t*  mat_inds
                  , ElTp*      mat_vals
                  , ElTp*      vec
                  , ElTp*      res
                  , const uint32_t num_rows
) {
  const int i  = blockIdx.x * blockDim.x + threadIdx.x;

  // check bounds
  if( i >= num_rows )
    return;

  ElTp acc = 0.0f;

  const int beg_row = rows[i];
  const int end_row = rows[i+1];
  for(int j = beg_row; j < end_row; j++) {
    const int col_ind = mat_inds[j];
    acc += mat_vals[j] * vec[col_ind];
  }
  res[i] = acc;
}

template<class ElTp, int blockSize>
__device__
void blockReduce(volatile ElTp* sdata) {
    int tid = threadIdx.x;
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
        if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
        if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
        if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
    }
    //if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class ElTp, int B> __global__ 
void spmvInnParKer( uint64_t*  rows
                  , uint32_t*  mat_inds
                  , ElTp* mat_vals
                  , ElTp* vec
                  , ElTp* res
                  , const uint32_t num_rows
) {
  extern __shared__ char sh_mem[];
  volatile ElTp* shmem = (ElTp*)sh_mem;

  const int i  = blockIdx.x;

  // all threads read the start/end of this row.
  uint64_t beg_row = rows[i] + threadIdx.x;
  uint64_t end_row = rows[i+1];

  // per-thread accumulation
  ElTp acc = 0;
  for(uint64_t j = beg_row; j < end_row; j+=blockDim.x) {
      const int  col_ind = mat_inds[j];
      const ElTp elm_val = mat_vals[j];
      acc += elm_val * vec[col_ind];
  }

  // reduction across Cuda-block threads
  shmem[threadIdx.x] = acc;
  
  __syncthreads();
  blockReduce<ElTp, B>(shmem);
  __syncthreads();

  if(threadIdx.x == 0) {
    res[i] = shmem[0];
  }
}

#endif
