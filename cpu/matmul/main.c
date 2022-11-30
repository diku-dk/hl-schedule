#include <stdio.h>
#include <CL/cl.h>

#include "goldenSeq.h"

#define OPENCL_SAFE_CALL(call, msg) {   \
    cl_int __err = call;                \
    if (err < 0) {                      \
        printf("%s: %d\n", msg, __err); \
        exit(1);                        \
    }                                   \
}

int main() {
    const int M = 1;
    const int N = 1000;
    const int K = 2048;

    float *a = (float *) malloc(M * K * sizeof(float));
    for (int i = 0; i < M * K; ++i) a[i] = (i % 10) + 1;
    float *b = (float *) malloc(K * N * sizeof(float));
    for (int i = 0; i < K * N; ++i) b[i] = (i % 10) + 1;
    float *c = (float *) malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; ++i) c[i] = 0;

    float *c_gold = (float *) malloc(M * N * sizeof(float));
    goldenSeq(a, b, c_gold, M, N, K);

    cl_int err;

    cl_platform_id platform;
    OPENCL_SAFE_CALL(clGetPlatformIDs(1, &platform, NULL), "Couldn't find any platforms");

    cl_device_id device;
    OPENCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL), "Couldn't find any devices");

    cl_context context;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    OPENCL_SAFE_CALL(err, "Couldn't create a context");

    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    cl_kernel kernel;
    program_handle = fopen("kernels.cl", "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *) malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    cl_program program;
    program = clCreateProgramWithSource(context, 1, (const char **) &program_buffer, &program_size, &err);
    OPENCL_SAFE_CALL(err, "Couldn't create the program");
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {

        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    kernel = clCreateKernel(program, "matmul", &err);
    OPENCL_SAFE_CALL(err, "Couldn't create the kernel");

    cl_mem a_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M * K * sizeof(float), a, &err);
    OPENCL_SAFE_CALL(err, "Couldn't create buffer a");
    cl_mem b_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K * N * sizeof(float), b, &err);
    OPENCL_SAFE_CALL(err, "Couldn't create buffer b");
    cl_mem c_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, M * N * sizeof(float), c, &err);
    OPENCL_SAFE_CALL(err, "Couldn't create buffer c");

    OPENCL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buff), "Couldn't set kernel argument a");
    OPENCL_SAFE_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buff), "Couldn't set kernel argument b");
    OPENCL_SAFE_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buff), "Couldn't set kernel argument c");

    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    OPENCL_SAFE_CALL(err, "Couldn't create the command queue");

    cl_event event = clCreateUserEvent(context, &err); OPENCL_SAFE_CALL(err, "Couldn't create event");
    cl_ulong start, end;
    size_t global_size[3] = {125, 1, 1};
    size_t local_size[3] = {1, 1, 1};
    for (int warmup = 0; warmup < 10; ++warmup) {
        OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size, local_size, 0, NULL, &event),
                         "Couldn't enqueue the kernel execution command");
        OPENCL_SAFE_CALL(clWaitForEvents(1, &event), "Error while waiting for kernel event");
    }
    cl_ulong min_runtime = CL_ULONG_MAX;
    for (int evaluation = 0; evaluation < 200; ++evaluation) {
        OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size, local_size, 0, NULL, &event),
                         "Couldn't enqueue the kernel execution command");
        OPENCL_SAFE_CALL(clWaitForEvents(1, &event), "Error while waiting for kernel event");
        OPENCL_SAFE_CALL(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL),
                         "Couldn't get start time");
        OPENCL_SAFE_CALL(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL),
                         "Couldn't get end time");
        cl_ulong runtime = end - start;
        if (runtime < min_runtime)
            min_runtime = runtime;
    }

    OPENCL_SAFE_CALL(clEnqueueReadBuffer(queue, c_buff, CL_TRUE, 0, M * N * sizeof(float), c, 0, NULL, NULL),
                     "Couldn't read kernel result");
    for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
        if (c[m * N + n] != c_gold[m * N + n]) {
            printf("incorrect result at index %d (expected %.0f, actual %.0f)\n",
                   m * N + n, c_gold[m * N + n], c[m * N + n]);
            exit(1);
        }
    printf("result is correct\n");
    printf("runtime: %lu ns\n", min_runtime);

    clReleaseEvent(event);
    clReleaseMemObject(a_buff);
    clReleaseMemObject(b_buff);
    clReleaseMemObject(c_buff);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    free(a);
    free(b);
    free(c);
}