#include <iostream>
#include <fstream>
#include <limits>

#include <cuda.h>
#include <nvrtc.h>

#include "goldenSeq.h"

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

int main() {
    const int N = 1;
    const int P = 112;
    const int Q = 112;
    const int K = 64;
    const int C = 3;
    const int R = 7;
    const int S = 7;

    float *images = (float *) malloc(N * (2 * P + R - 1) * (2 * Q + S - 1) * C * sizeof(float));
    for (int i = 0; i < N * (2 * P + R - 1) * (2 * Q + S - 1) * C; ++i) images[i] = (i % 10) + 1;
    float *filter = (float *) malloc(K * R * S * C * sizeof(float));
    for (int i = 0; i < K * R * S * C; ++i) filter[i] = (i % 10) + 1;
    float *out = (float *) malloc(N * P * Q * K * sizeof(float));
    for (int i = 0; i < N * P * Q * K; ++i) out[i] = 0;

    float *out_gold = (float *) malloc(N * P * Q * K * sizeof(float));
    goldenSeq(images, filter, out_gold, N, P, Q, K, C, R, S);

    std::ifstream kernel_stream("kernels.cuda");
    std::string mcc_kernel = std::string(std::istreambuf_iterator<char>(kernel_stream),
                                         std::istreambuf_iterator<char>());
    kernel_stream.close();

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, mcc_kernel.c_str(), "kernels.cuda", 0, nullptr, nullptr));
    nvrtcResult compileResult = nvrtcCompileProgram(prog, 0, nullptr);
    if (compileResult != NVRTC_SUCCESS) {
        exit(1);
    }
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "convolution"));

    CUdeviceptr dimages, dfilter, dout;
    CUDA_SAFE_CALL(cuMemAlloc(&dimages, N * (2 * P + R - 1) * (2 * Q + S - 1) * C * sizeof(float)));
    CUDA_SAFE_CALL(cuMemAlloc(&dfilter, K * R * S * C * sizeof(float)));
    CUDA_SAFE_CALL(cuMemAlloc(&dout, N * P * Q * K * sizeof(float)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dimages, images, N * (2 * P + R - 1) * (2 * Q + S - 1) * C * sizeof(float)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dfilter, filter, K * R * S * C * sizeof(float)));

    CUevent start, end;
    CUDA_SAFE_CALL(cuEventCreate(&start, 0));
    CUDA_SAFE_CALL(cuEventCreate(&end, 0));
    void *args[] = {&dimages, &dfilter, &dout};
    for (int warmup = 0; warmup < 10; ++warmup) {
        CUDA_SAFE_CALL(
            cuLaunchKernel(kernel,
                           2 * 7 * 7, 1, 1,
                           16 * 4 * 4, 1, 1,
                           0, nullptr,
                           args, nullptr));
        CUDA_SAFE_CALL(cuCtxSynchronize());
    }
    size_t min_runtime = std::numeric_limits<size_t>::max();
    for (int evaluation = 0; evaluation < 200; ++evaluation) {
        CUDA_SAFE_CALL(cuEventRecord(start, nullptr));
        CUDA_SAFE_CALL(
            cuLaunchKernel(kernel,
                           2 * 7 * 7, 1, 1,
                           16 * 4 * 4, 1, 1,
                           0, nullptr,
                           args, nullptr));
        CUDA_SAFE_CALL(cuEventRecord(end, nullptr));
        CUDA_SAFE_CALL(cuCtxSynchronize());

        float runtime_ms;
        CUDA_SAFE_CALL(cuEventElapsedTime(&runtime_ms, start, end));
        size_t runtime = runtime_ms * 1000000;
        if (runtime < min_runtime)
            min_runtime = runtime;
    }

    CUDA_SAFE_CALL(cuMemcpyDtoH(out, dout, N * P * Q * K * sizeof(float)));
    for (int n = 0; n < N; ++n)
    for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
    for (int k = 0; k < K; ++k)
        if (out[n * P * Q * K + p * Q * K + q * K + k] != out_gold[n * P * Q * K + p * Q * K + q * K + k]) {
            printf("incorrect result at index %d (expected %.0f, actual %.0f)\n",
                   n * P * Q * K + p * Q * K + q * K + k,
                   out_gold[n * P * Q * K + p * Q * K + q * K + k],
                   out[n * P * Q * K + p * Q * K + q * K + k]);
            exit(1);
        }
    printf("result is correct\n");
    printf("runtime: %lu ns\n", min_runtime);

    CUDA_SAFE_CALL(cuMemFree(dimages));
    CUDA_SAFE_CALL(cuMemFree(dfilter));
    CUDA_SAFE_CALL(cuMemFree(dout));
    CUDA_SAFE_CALL(cuModuleUnload(module));
    CUDA_SAFE_CALL(cuCtxDestroy(context));
    delete[] ptx;
    free(images);
    free(filter);
    free(out);
    free(out_gold);
}
