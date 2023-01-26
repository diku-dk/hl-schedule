#include "../../../helper.h"

#define ERR         0.000000001

void goldenSeq(float* images, float* filter, float* out,
               const int N, const int P, const int Q, const int K, const int C, const int R, const int S) {
    for (int n = 0; n < N; ++n) {
    for (int p = 0; p < P; ++p) {
    for (int q = 0; q < Q; ++q) {
    for (int k = 0; k < K; ++k) {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
            acc +=  images[ n * (2 * P + R - 1) * (2 * Q + S - 1) * C +
                            c * (2 * P + R - 1) * (2 * Q + S - 1) +
                            (2 * p + r) * (2 * Q + S - 1) +
                            2 * q + s
                          ] *
                    filter[c * R * S * K + r * S * K + s * K + k];
        }}}
        out[n * P * Q * K + p * Q * K + q * K + k] = acc;
    }}}}
}


template <class ElTp, int Tpq, int Rpq, int Tk, int Rk, int Trs>
__global__ void convolutionKer(ElTp* images, ElTp* filter, ElTp* out, int N, int P, int Q, int K, int C, int R, int S) {
  __shared__ ElTp images_sh[2*Tpq*Rpq + Trs - 2][2*Tpq*Rpq + Trs - 2 + 1];
  __shared__ ElTp filter_sh[Trs][Tk*Rk];
  ElTp acc[Rpq][Rpq][Rk];

  // i am ignoring n
  uint32_t gid = threadIdx.z * (Tpq * Tk) + threadIdx.y * Tk + threadIdx.x;

  //Assuming N = 1 and n = 0
  uint32_t pp = blockIdx.z * (Tpq * Rpq);
  uint32_t qq = blockIdx.y * (Tpq * Rpq);
  uint32_t kk = blockIdx.x * (Tk * Rk);
  
  // init regs
  #pragma unroll
  for(int p=0; p<Rpq; p++)
  #pragma unroll
  for(int q=0; q<Rpq; q++)
  #pragma unroll
  for(int k=0; k<Rk; k++)
    acc[p][q][k] = 0.0;

  for(int c=0; c<C; c++) {
    for(int rr=0; rr<R; rr+=Trs) {
      for(int ss=0; ss<S; ss+=Trs) {

        // copy the slice of images from global to shared mem
        for(    uint32_t ind = gid;
                ind < (2*Tpq*Rpq + Trs - 2)*(2*Tpq*Rpq + Trs - 2); 
                ind += Tpq*Tpq*Tk
        ) {
            ElTp el = 0;
            uint32_t ind_y = ind / (2*Tpq*Rpq + Trs - 2);
            uint32_t ind_x = ind - ind_y * (2*Tpq*Rpq + Trs - 2);
            if( (pp + rr + ind_y < 2*P+R-1) && (qq + ss + ind_x < 2*Q+S-1) ) {
              el = images[ c*(2*P+R-1)*(2*Q+S-1) + (2*pp+rr+ind_y)*(2*Q+S-1) + (2*qq+ss+ind_x) ];
            }
            images_sh[ind_y][ind_x] = el;
        }

        //#pragma unroll
        for(int r = 0; r < Trs; r++) {
            // copy the slice of filter from global to shared mem
            for( uint32_t ind = gid; ind < Tk*Rk*Trs; ind += Tpq*Tpq*Tk ) {
                ElTp el = 0;

                uint32_t ind_s = ind / (Tk*Rk);
                uint32_t ind_k = ind - ind_s * (Tk*Rk);

                if(kk+ind_k < K && ss + ind_s<S && rr+r < R) {
                  el = filter[c*(R*S*K) + (rr+r)*(S*K) + (ss + ind_s)*K + kk + ind_k];
                }
                filter_sh[ind_s][ind_k] = el;
            }
            __syncthreads();

            // finally the computation:
            #pragma unroll
            for(uint32_t s = 0; s < Trs; s++)
            #pragma unroll
            for(uint32_t p = 0; p < Rpq; p++)
            #pragma unroll
            for(uint32_t q = 0; q < Rpq; q++)
            #pragma unroll
            for(uint32_t k = 0; k < Rk;  k++) {
                acc[p][q][k] += images_sh[2*(threadIdx.z*Rpq + p)+r][2*(threadIdx.y*Rpq + q)+s] * 
                                      // [2*Tpq*Rpq + Trs - 2][2*Tpq*Rpq + Trs - 2];
                                filter_sh[s][threadIdx.x*Rk+k];
            }
            __syncthreads();
        }

      }
    }
  }

  // write to global memory
  #pragma unroll
  for(uint32_t p = 0; p < Rpq; p++)
  #pragma unroll
  for(uint32_t q = 0; q < Rpq; q++)
  #pragma unroll
  for(uint32_t k = 0; k < Rk;  k++) {
    uint32_t ind_p = pp + p + threadIdx.z*Rpq;
    uint32_t ind_q = qq + q + threadIdx.y*Rpq;
    uint32_t ind_k = kk + k + threadIdx.x*Rk;
    if( (ind_p < P) && (ind_q < Q) && (ind_k < K) ) {
        out[ind_p*Q*K + ind_q*K + ind_k] = acc[p][q][k];
    }
  }
}


template <class ElTp, int Tpq, int Rpq, int Tk, int Rk, int Trs>
__global__ void convolutionKer2(ElTp* images, ElTp* filter, ElTp* out, int N, int P, int Q, int K, int C, int R, int S) {
  __shared__ ElTp images_sh[2*Tpq*Rpq + Trs - 2][2*Tpq*Rpq + Trs - 2 + 1];
  __shared__ ElTp filter_sh[Trs][Trs][Tk*Rk];
  ElTp acc[Rpq][Rpq][Rk];

  // i am ignoring n
  uint32_t gid = threadIdx.z * (Tpq * Tk) + threadIdx.y * Tk + threadIdx.x;

  //Assuming N = 1 and n = 0
  uint32_t pp = blockIdx.z * (Tpq * Rpq);
  uint32_t qq = blockIdx.y * (Tpq * Rpq);
  uint32_t kk = blockIdx.x * (Tk * Rk);
  
  // init regs
  #pragma unroll
  for(int p=0; p<Rpq; p++)
  #pragma unroll
  for(int q=0; q<Rpq; q++)
  #pragma unroll
  for(int k=0; k<Rk; k++)
    acc[p][q][k] = 0.0;

  for(int c=0; c<C; c++) {
    for(int rr=0; rr<R; rr+=Trs) {
      for(int ss=0; ss<S; ss+=Trs) {

        // copy the slice of images from global to shared mem
        for(    uint32_t ind = gid;
                ind < (2*Tpq*Rpq + Trs - 2)*(2*Tpq*Rpq + Trs - 2); 
                ind += Tpq*Tpq*Tk
        ) {
            ElTp el = 0;
            uint32_t ind_y = ind / (2*Tpq*Rpq + Trs - 2);
            uint32_t ind_x = ind - ind_y * (2*Tpq*Rpq + Trs - 2);
            if( (pp + rr + ind_y < 2*P+R-1) && (qq + ss + ind_x < 2*Q+S-1) ) {
              el = images[ c*(2*P+R-1)*(2*Q+S-1) + (2*pp+rr+ind_y)*(2*Q+S-1) + (2*qq+ss+ind_x) ];
            }
            images_sh[ind_y][ind_x] = el;
        }
        
        // copy the slice of filter from global to shared mem
        for( uint32_t ind = gid; ind < Tk*Rk*Trs*Trs; ind += Tpq*Tpq*Tk ) {
            ElTp el = 0;

            uint32_t ind_sr = ind / (Tk*Rk);
            uint32_t ind_k = ind - ind_sr * (Tk*Rk);
            uint32_t ind_r = ind_sr / Trs;
            uint32_t ind_s = ind_sr - ind_r*Trs;

            if(kk+ind_k < K && ss + ind_s<S && rr+ind_r < R) {
              el = filter[c*(R*S*K) + (rr+ind_r)*(S*K) + (ss + ind_s)*K + kk + ind_k];
            }
            filter_sh[ind_r][ind_s][ind_k] = el;
        }
        __syncthreads();
            
        // finally the computation:
        #pragma unroll
        for(int r = 0; r < Trs; r++)
        #pragma unroll
        for(uint32_t s = 0; s < Trs; s++)
        #pragma unroll
        for(uint32_t p = 0; p < Rpq; p++)
        #pragma unroll
        for(uint32_t q = 0; q < Rpq; q++)
        #pragma unroll
        for(uint32_t k = 0; k < Rk;  k++) {
            acc[p][q][k] += images_sh[2*(threadIdx.z*Rpq + p)+r][2*(threadIdx.y*Rpq + q)+s] * 
                                  // [2*Tpq*Rpq + Trs - 2][2*Tpq*Rpq + Trs - 2];
                            filter_sh[r][s][threadIdx.x*Rk+k];
        }
        __syncthreads();
      }
    }
  }

  // write to global memory
  #pragma unroll
  for(uint32_t p = 0; p < Rpq; p++)
  #pragma unroll
  for(uint32_t q = 0; q < Rpq; q++)
  #pragma unroll
  for(uint32_t k = 0; k < Rk;  k++) {
    uint32_t ind_p = pp + p + threadIdx.z*Rpq;
    uint32_t ind_q = qq + q + threadIdx.y*Rpq;
    uint32_t ind_k = kk + k + threadIdx.x*Rk;
    if( (ind_p < P) && (ind_q < Q) && (ind_k < K) ) {
        out[ind_p*Q*K + ind_q*K + ind_k] = acc[p][q][k];
    }
  }
}


int main() {
    const int N = 1;
    const int P = 112;
    const int Q = 112;
    const int K = 64;
    const int C = 3;
    const int R = 7;
    const int S = 7;

    // 1. allocate and init host array
    size_t size_images = N * (2 * P + R - 1) * (2 * Q + S - 1) * C * sizeof(float);
    float *images = (float *) malloc(size_images);
    for (int i = 0; i < N * (2 * P + R - 1) * (2 * Q + S - 1) * C; ++i) images[i] = (i % 10) + 1;

    size_t size_filter = K * R * S * C * sizeof(float);
    float *filter = (float *) malloc(size_filter);
    for (int i = 0; i < K * R * S * C; ++i) filter[i] = (i % 10) + 1;

    size_t size_out = N * P * Q * K * sizeof(float);
    float *out = (float *) malloc(size_out);
    float *out_ref = (float *) malloc(size_out);
    for (int i = 0; i < N * P * Q * K; ++i) out_ref[i] = 0;

    // 2. golden sequential on host
    goldenSeq(images, filter, out_ref, N, P, Q, K, C, R, S);

    // 3. allocate device memory
    float* d_images;
    float* d_filter;
    float* d_out;
    cudaMalloc((void**) &d_images, size_images);
    cudaMalloc((void**) &d_filter, size_filter);
    cudaMalloc((void**) &d_out, size_out);
 
    // 4. copy host memory to device
    cudaMemcpy(d_images, images, size_images, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, size_filter, cudaMemcpyHostToDevice);

    // 5. dry run
    const int Tpq = 4;
    const int Rpq = 3;
    const int Tk  = 8;
    const int Rk  = 4;
    int  dimz = ceil( ((float) P)/(Tpq*Rpq) );
    int  dimy = ceil( ((float) Q)/(Tpq*Rpq) ); 
    int  dimx = ceil( ((float) K)/(Tk*Rk) );
    
    dim3 block(Tk, Tpq, Tpq);
    dim3 grid(dimx, dimy, dimz);
    convolutionKer2<float, Tpq, Rpq, Tk, Rk, 7><<< grid, block >>>(d_images, d_filter, d_out, N, P, Q, K, C, R, S);
    cudaDeviceSynchronize();

    // time measurement
    {
        const int GPU_RUNS = 100;
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
      
        for(int i=0; i<GPU_RUNS; i++) {
            convolutionKer2<float, Tpq, Rpq, Tk, Rk, 7><<< grid, block >>>(d_images, d_filter, d_out, N, P, Q, K, C, R, S);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

        gpuAssert( cudaPeekAtLastError() );

        float microsecPerMatrixMul = elapsed; 
        double flopsPerMatrixMul = N * P * Q * K * (C * R * S + C * R * S - 1); 
        double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

        printf( "Convolution first attempt runs in: %lu microsecs, GFlops/sec: %.2f (but without accounting for transpositions)\n"
              , elapsed, gigaFlops );

        // copy result from device to host
        cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost);
       
        validate<float>(out_ref, out, size_out/sizeof(float), ERR);
    }

    free(images);
    free(filter);
    free(out);
    free(out_ref);
    cudaFree(d_images);
    cudaFree(d_filter);
    cudaFree(d_out);

    return 0;
}