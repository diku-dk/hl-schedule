#include "../helper.h"
#include "kernels.cu.h"
#include "goldenSeq.h"

using namespace std;

#define GPU_RUNS    300
#define ERR         0.000005


uint64_t
computeRowStarts( uint64_t* h_rows
                , const int num_rows
                , const int vec_len
) {
    uint64_t flat_len = 0;
    h_rows[0] = 0;
    for(int i=1; i<=num_rows; i++) {
        float r = rand() / (float)RAND_MAX;
        uint32_t len = ceil(r * vec_len);
        len = (len <= 0) ? 1 : len;
        flat_len += len;
        h_rows[i] = flat_len;
    }
    return flat_len;
}

void initColInds(uint32_t* h_mat_inds, const uint64_t flat_len, const uint32_t vec_len) {
    for(uint64_t i=0; i<flat_len; i++) {
        float r = rand() / (float)RAND_MAX;
        uint32_t ind = floor(r*vec_len);
        h_mat_inds[i] = ind;
    }
}

template<class ElTp>
void runNaive ( uint64_t*  d_rows
              , uint32_t*  d_mat_inds
              , ElTp*      d_mat_vals
              , ElTp*      d_vec
              , ElTp*      d_res
              , ElTp*      h_ref
              , uint32_t   num_rows
              , uint64_t   flat_len
) {
    unsigned long long mem_size_res = num_rows * sizeof(ElTp);
    cudaMemset (d_res, 0, mem_size_res );

    // setup execution parameters
    const int  block= 128;
    const int  grid = (num_rows + block - 1) / block;

    // dry run
    spmvNaiveKer<ElTp><<< grid, block >>>( d_rows, d_mat_inds, d_mat_vals, d_vec, d_res, num_rows );
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    
    for(int i=0; i<GPU_RUNS; i++) {
        spmvNaiveKer<ElTp><<< grid, block >>>( d_rows, d_mat_inds, d_mat_vals, d_vec, d_res, num_rows );
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    float microsecs = elapsed; 
    double bytesAccessed = flat_len * ( 2*sizeof(ElTp) + sizeof(uint32_t) );
    double gigaBytesSec = (bytesAccessed * 1.0e-3f) / microsecs; 

    printf("GPU Naive SpMatVec version runs in: %lu microsecs, GB/sec: %.2f\n", elapsed, gigaBytesSec);

    ElTp* h_res = (ElTp*) malloc(mem_size_res);
    cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost);
    validate<ElTp>(h_ref, h_res, num_rows, ERR);
    free(h_res);
}

template<class ElTp, int B>
void runOptim ( uint64_t*  d_rows
              , uint32_t*  d_mat_inds
              , ElTp*      d_mat_vals
              , ElTp*      d_vec
              , ElTp*      d_res
              , ElTp*      h_ref
              , uint32_t   num_rows
              , uint64_t   flat_len
) {
    unsigned long long mem_size_res = num_rows * sizeof(ElTp);
    cudaMemset (d_res, 0, mem_size_res );

    // setup execution parameters
    const int  block= B;
    const int  grid = num_rows;

    // dry run
    spmvInnParKer<ElTp,B><<< grid, block, block*sizeof(ElTp) >>>( d_rows, d_mat_inds, d_mat_vals, d_vec, d_res, num_rows );
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    
    for(int i=0; i<GPU_RUNS; i++) {
        spmvInnParKer<ElTp,B><<< grid, block, block*sizeof(ElTp) >>>( d_rows, d_mat_inds, d_mat_vals, d_vec, d_res, num_rows );
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    float microsecs = elapsed; 
    double bytesAccessed = flat_len * ( 2*sizeof(ElTp) + sizeof(uint32_t) );
    double gigaBytesSec = (bytesAccessed * 1.0e-3f) / microsecs; 

    printf("GPU Inner-Parallel SpMatVec version runs in: %lu microsecs, GB/sec: %.2f\n", elapsed, gigaBytesSec);

    ElTp* h_res = (ElTp*) malloc(mem_size_res);
    cudaMemcpy(h_res, d_res, mem_size_res, cudaMemcpyDeviceToHost);
    validate<ElTp>(h_ref, h_res, num_rows, ERR);
    free(h_res);
}

/**
 * This will run all code versions
 * (and summarize the performance in GFlops). 
 */
template<class ElTp, int B>
void runAll ( const uint32_t vec_len
            , const uint32_t num_rows
) {

    srand(2022);

    // 1. allocate exclusive-scanned shape "h_rows", and compute it
    uint64_t  mem_size_rows = (num_rows+1) * sizeof(uint64_t);
    uint64_t* h_rows = (uint64_t*) malloc( mem_size_rows );
    const uint64_t flat_len = computeRowStarts( h_rows, num_rows, vec_len );
 
    // 1. allocate host memory
    uint64_t  mem_size_vec= vec_len * sizeof(ElTp);
    ElTp*     h_vec = (ElTp*) malloc( mem_size_vec );

    uint64_t  mem_size_res= num_rows* sizeof(ElTp);
    ElTp*     h_res = (ElTp*) malloc( mem_size_res );

    uint64_t  mem_size_inds = flat_len * sizeof(uint32_t);
    uint32_t* h_mat_inds = (uint32_t*) malloc( mem_size_inds );

    uint64_t  mem_size_vals = flat_len * sizeof(ElTp);
    ElTp*     h_mat_vals = (ElTp*) malloc( mem_size_vals );
 
    // 2. initialize input arrays in host memory
    randomInit<ElTp>(h_vec, vec_len);
    randomInit<ElTp>(h_mat_vals, flat_len);
    initColInds(h_mat_inds, flat_len, vec_len);
    
    
    // 3. allocate device memory
    uint64_t *d_rows;
    uint32_t* d_mat_inds;
    ElTp *d_mat_vals, *d_vec, *d_res;
    cudaMalloc((void**) &d_rows, mem_size_rows );
    cudaMalloc((void**) &d_vec,  mem_size_vec  );
    cudaMalloc((void**) &d_res,  mem_size_res  );
    cudaMalloc((void**) &d_mat_inds, mem_size_inds );
    cudaMalloc((void**) &d_mat_vals, mem_size_vals );
 
    // 4. copy host memory to device
    cudaMemcpy(d_rows, h_rows, mem_size_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec,  h_vec,  mem_size_vec,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_res,  h_res,  mem_size_res,  cudaMemcpyHostToDevice);

    cudaMemcpy(d_mat_inds, h_mat_inds, mem_size_inds, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_vals, h_mat_vals, mem_size_vals, cudaMemcpyHostToDevice);

  
    printf("Sizes are: (num-rows, vec-len, flat_len, cuda-block)=(%d, %d, %llu, %d)\n"
          , num_rows, vec_len, flat_len, B );

    // 5. compute golden sequential
    goldenSeq<ElTp>( h_rows, h_mat_inds, h_mat_vals, h_vec, h_res, num_rows );

    // 6. compute the naive GPU version
    runNaive<ElTp>( d_rows, d_mat_inds, d_mat_vals, d_vec, d_res, h_res, num_rows, flat_len );

    // 7. compute the register-tiled version
    runOptim<ElTp,B>( d_rows, d_mat_inds, d_mat_vals, d_vec, d_res, h_res, num_rows, flat_len );

    free(h_rows);
    free(h_vec);
    free(h_res);
    free(h_mat_inds);
    free(h_mat_vals);
    cudaFree(d_rows);
    cudaFree(d_vec);
    cudaFree(d_res);
    cudaFree(d_mat_inds);
    cudaFree(d_mat_vals);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main (int argc, char * argv[]) {
    if (argc != 3) {
        printf("Usage: %s vct-len num-rows\n", argv[0]);
        exit(1);
    }
    const uint32_t vct_len  = atoi(argv[1]);
    const uint32_t num_rows = atoi(argv[2]);

    runAll<float,256> ( vct_len, num_rows );
    runAll<double,256> ( vct_len, num_rows );
}
