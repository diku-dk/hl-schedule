/**
 * This NW Cuda implementation is essentially
 * taken from Rodinia_3.1 suite and slightly modified, e.g.,
 * ToDos:
 *   1. fix for large matrix, i.e., indices int => long long
 *   2. eliminate conflicts to shared memory banks,
 */

#define GPU_RUNS   500
#define BLOCK_SIZE 32

// includes, kernels
#include "../helper.h"
#include "kernels.cu.h"
#include "goldenSeq.h"

int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}


void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void initInputSeq(int* input_itemsets, const int N, const int penalty) {
    for( int i = 1; i< N ; i++)
       input_itemsets[i*N] = -i * penalty;
	for( int j = 1; j< N ; j++)
       input_itemsets[j] = -j * penalty;
}

void initInputCuda(int* input_d, const int N, const int penalty) {
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
	int num_blocks = ( N - 1 ) / BLOCK_SIZE;
    dim3 dimGrid(num_blocks,  1, 1);
    initInputKer<<<dimGrid, dimBlock>>>(input_d, N, penalty); 
}

void runCuda(int* matrix_cuda, int* referrence_cuda, const int N, const int penalty) {
    dim3 dimGrid;
	dim3 dimBlock(BLOCK_SIZE, 1);
	int block_width = ( N - 1 ) / BLOCK_SIZE;

    // re-init since the algorithm is in place
    initInputCuda(matrix_cuda, N, penalty);

	//process top-left matrix matrix
	for( int i = 1 ; i <= block_width ; i++ ) {
		dimGrid.x = i;
		dimGrid.y = 1;
		needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence_cuda, matrix_cuda
		                                           , N, penalty, i, block_width ); 
	}

    //process bottom-right matrix
	for( int i = block_width - 1  ; i >= 1 ; i-- ) {
		dimGrid.x = i;
		dimGrid.y = 1;
		needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence_cuda, matrix_cuda
		                                           , N, penalty, i, block_width ); 
	}
}

int main( int argc, char** argv) 
{
    int N, penalty;
    int *input_itemsets, *output_itemsets, *referrence;
	int *matrix_cuda,  *referrence_cuda;
	uint64_t size;
	
    printf("Cuda block size of kernel = %d \n", BLOCK_SIZE);    

    // 1. Reading dimensions;
    //    assumes an N x N matrix
	if (argc == 3) {
		N = atoi(argv[1]);
		penalty = atoi(argv[2]);
	} else {
	    usage(argc, argv);
    }
	
	if( (N < 2) || ((N % BLOCK_SIZE) != 0) ) {
	    fprintf(stderr,"The dimension values must be a multiple of BLOCK_SIZE\n");
	    exit(1);
	}
	

	N = N + 1;
	referrence = (int *)malloc( N * N * sizeof(int) );
    input_itemsets = (int *)malloc( N * N * sizeof(int) );
	output_itemsets = (int *)malloc( N * N * sizeof(int) );
	

	if (!input_itemsets) {
		fprintf(stderr, "error: can not allocate memory");
        exit(1);
    }

    // 2. Initializing input and reference arrays
    srand ( 7 );
	
    for (int i = 0 ; i < N; i++){
		for (int j = 0 ; j < N; j++){
			input_itemsets[i*N+j] = 0;
		}
	}

	for( int i=1; i<N; i++){    //please define your own sequence. 
       input_itemsets[i*N] = rand() % 10 + 1;
	}

    for( int j=1; j<N; j++){    //please define your own sequence.
       input_itemsets[j] = rand() % 10 + 1;
	}

	for (int i = 1 ; i < N; i++){
		for (int j = 1 ; j < N; j++){
		referrence[i*N+j] = blosum62[input_itemsets[i*N]][input_itemsets[j]];
		}
	}

    initInputSeq(input_itemsets, N, penalty);

    // 3. Allocation and initialization of Cuda buffers
    size = (uint64_t)N;
    size = size * size;

	cudaError_t failed1 = cudaMalloc((void**)& referrence_cuda, sizeof(int)*size);
	cudaError_t failed2 = cudaMalloc((void**)& matrix_cuda, sizeof(int)*size);

	cudaMemcpy(referrence_cuda, referrence, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size, cudaMemcpyHostToDevice);

    gpuAssert( cudaPeekAtLastError() );

    if(failed1 != 0 || failed2 != 0) {
        printf("Cuda allocation failed, allocation size: %llu, Exiting!\n", 2*size*sizeof(int));
        exit(1);
    }

    // 4. Run golden sequential
    goldenSeq( input_itemsets, referrence, N, penalty );

    // 5. Run Cuda implementation
    {
        // dry run
        runCuda(matrix_cuda, referrence_cuda, N, penalty);

        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<GPU_RUNS; i++) {
            runCuda(matrix_cuda, referrence_cuda, N, penalty);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
        gpuAssert( cudaPeekAtLastError() );

        double M = N-1;
        double bytes = 2*M*sizeof(int) + M*M*sizeof(int)*5;
        double gigaBytesPerSec = (bytes * 1.0e-3f) / elapsed; 

        printf("Cuda implementation of NW of size (%d x %d) runs in %lu microsecs, GFlops/sec: %.2f\n"
              , N, N, elapsed, gigaBytesPerSec);
    }

    // 6. Validate Cuda result vs golden sequential
    cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size, cudaMemcpyDeviceToHost);
	validateExact(input_itemsets, output_itemsets, size);

    // 7. Free host and device memory
	cudaFree(referrence_cuda);
	cudaFree(matrix_cuda);

	free(referrence);
	free(input_itemsets);
	free(output_itemsets);
	
    return 0;
}

