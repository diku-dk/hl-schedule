extern "C" __global__
void matmul(float const * const __restrict__ a_buf_raw, float const * const __restrict__ b_buf_raw, float * const __restrict__ int_res_c_raw) {
    size_t l_cb_offset_l_1, l_cb_offset_l_2, l_cb_offset_r_1, p_cb_offset_l_1, p_cb_offset_l_2, p_cb_offset_r_1;

    const size_t i_wg_l_1 = 0;
    const size_t i_wi_l_1 = 0;
    const size_t i_wg_l_2 = blockIdx.x;
    const size_t i_wi_l_2 = threadIdx.x;
    const size_t i_wg_r_1 = 0;
    const size_t i_wi_r_1 = threadIdx.y;

    float const (*a_buf)[2048] = reinterpret_cast<float const (*)[2048]>(a_buf_raw);
    float const (*b_buf)[1000] = reinterpret_cast<float const (*)[1000]>(b_buf_raw);
    float (*int_res_c)[(((1000 - 1)) - (0) + 1)] = reinterpret_cast<float (*)[(((1000 - 1)) - (0) + 1)]>(int_res_c_raw);
    __shared__ float c_l_reduction_mem[1][32][20];
    float res_p_c[((1 / 1) / (1)) + (((1 * (1 / 1)) % (1 * (1)) / 1) > 0) + (((1 * (1 / 1)) % (1 * (1)) % 1) > 0)][1][1][((20 / 20) / (1)) + (((20 * (20 / 20)) % (20 * (1)) / 20) > 0) + (((20 * (20 / 20)) % (20 * (1)) % 20) > 0)][1];

    l_cb_offset_l_1 = i_wg_l_1 * 1;
    __syncthreads();
    for (size_t l_step_l_1 = 0; l_step_l_1 < ((1 / (1 * 1)) / (1 / 1)); ++l_step_l_1) {
        __syncthreads();
        l_cb_offset_l_2 = i_wg_l_2 * 20;
        for (size_t l_step_l_2 = 0; l_step_l_2 < ((1000 / (50 * 20)) / (20 / 20)); ++l_step_l_2) {
            __syncthreads();
            l_cb_offset_r_1 = i_wg_r_1 * 32;
            size_t l_step_r_1 = 0;
            p_cb_offset_l_1 = i_wi_l_1 * 1;
            for (size_t p_step_l_1 = 0; p_step_l_1 < ((1 / 1) / (1)); ++p_step_l_1) {
                p_cb_offset_l_2 = i_wi_l_2 * 1;
                for (size_t p_step_l_2 = 0; p_step_l_2 < ((20 / 20) / (1)); ++p_step_l_2) {
                    p_cb_offset_r_1 = i_wi_r_1 * 1;
                    size_t p_step_r_1 = 0;
                    #pragma unroll
                    for (size_t p_iteration_l_1 = 0; p_iteration_l_1 < (1); ++p_iteration_l_1) {
                        #pragma unroll
                        for (size_t p_iteration_l_2 = 0; p_iteration_l_2 < (1); ++p_iteration_l_2) {
                            size_t p_iteration_r_1 = 0;
                            res_p_c[p_step_l_1][(p_iteration_l_1)][(0)][p_step_l_2][(p_iteration_l_2)] = a_buf[(l_cb_offset_l_1 + (((p_cb_offset_l_1 + (((p_iteration_l_1)) / 1) * 1 + 0)) / 1) * (1 * 1) + i_wi_l_1)][(l_cb_offset_r_1 + (((p_cb_offset_r_1 + (((p_iteration_r_1)) / 1) * 32 + 0)) / 32) * (1 * 32) + i_wi_r_1)] * b_buf[(l_cb_offset_r_1 + (((p_cb_offset_r_1 + (((p_iteration_r_1)) / 1) * 32 + 0)) / 32) * (1 * 32) + i_wi_r_1)][(l_cb_offset_l_2 + (((p_cb_offset_l_2 + (((p_iteration_l_2)) / 1) * 20 + 0)) / 20) * (50 * 20) + i_wi_l_2)];
                        }
                    }
                    p_cb_offset_r_1 += 32 * (1);
                    p_cb_offset_l_2 += 20 * (1);
                }
                p_cb_offset_l_1 += 1 * (1);
            }
            l_cb_offset_r_1 += (1 * 32) * (32 / 32);
            for (l_step_r_1 = 1; l_step_r_1 < ((2048 / (1 * 32)) / (32 / 32)); ++l_step_r_1) {
                p_cb_offset_l_1 = i_wi_l_1 * 1;
                for (size_t p_step_l_1 = 0; p_step_l_1 < ((1 / 1) / (1)); ++p_step_l_1) {
                    p_cb_offset_l_2 = i_wi_l_2 * 1;
                    for (size_t p_step_l_2 = 0; p_step_l_2 < ((20 / 20) / (1)); ++p_step_l_2) {
                        p_cb_offset_r_1 = i_wi_r_1 * 1;
                        size_t p_step_r_1 = 0;
                        #pragma unroll
                        for (size_t p_iteration_l_1 = 0; p_iteration_l_1 < (1); ++p_iteration_l_1) {
                            for (size_t p_iteration_l_2 = 0; p_iteration_l_2 < (1); ++p_iteration_l_2) {
                                for (size_t p_iteration_r_1 = 0; p_iteration_r_1 < (1); ++p_iteration_r_1) {
                                    res_p_c[p_step_l_1][(p_iteration_l_1)][(0)][p_step_l_2][(p_iteration_l_2)] += a_buf[(l_cb_offset_l_1 + (((p_cb_offset_l_1 + (((p_iteration_l_1)) / 1) * 1 + 0)) / 1) * (1 * 1) + i_wi_l_1)][(l_cb_offset_r_1 + (((p_cb_offset_r_1 + (((p_iteration_r_1)) / 1) * 32 + 0)) / 32) * (1 * 32) + i_wi_r_1)] * b_buf[(l_cb_offset_r_1 + (((p_cb_offset_r_1 + (((p_iteration_r_1)) / 1) * 32 + 0)) / 32) * (1 * 32) + i_wi_r_1)][(l_cb_offset_l_2 + (((p_cb_offset_l_2 + (((p_iteration_l_2)) / 1) * 20 + 0)) / 20) * (50 * 20) + i_wi_l_2)];
                                }
                            }
                        }
                        p_cb_offset_r_1 += 32 * (1);
                        p_cb_offset_l_2 += 20 * (1);
                    }
                    p_cb_offset_l_1 += 1 * (1);
                }
                l_cb_offset_r_1 += (1 * 32) * (32 / 32);
            }
            {
                __syncthreads();
                p_cb_offset_l_1 = i_wi_l_1 * 1;
                for (size_t p_step_l_1 = 0; p_step_l_1 < ((1 / 1) / (1)); ++p_step_l_1) {
                    p_cb_offset_l_2 = i_wi_l_2 * 1;
                    for (size_t p_step_l_2 = 0; p_step_l_2 < ((20 / 20) / (1)); ++p_step_l_2) {
                        for (size_t p_iteration_l_1 = 0; p_iteration_l_1 < (1); ++p_iteration_l_1) {
                            for (size_t p_iteration_l_2 = 0; p_iteration_l_2 < (1); ++p_iteration_l_2) {
                                {
                                    {
                                        c_l_reduction_mem[i_wi_l_1][i_wi_r_1][i_wi_l_2] = res_p_c[p_step_l_1][(p_iteration_l_1)][(0)][p_step_l_2][(p_iteration_l_2)];
                                    }
                                    res_p_c[p_step_l_1][(p_iteration_l_1)][(0)][p_step_l_2][(p_iteration_l_2)] = 0;
                                }
                                __syncthreads();
                                {
                                    size_t stride = (32) / 2;
                                    for (; stride > 0; stride /= 2) {
                                        if (i_wi_r_1 < stride) {
                                            c_l_reduction_mem[i_wi_l_1][i_wi_r_1][i_wi_l_2] += c_l_reduction_mem[i_wi_l_1][((i_wi_r_1) + stride)][i_wi_l_2];
                                        }
                                        __syncthreads();
                                    }
                                    __syncthreads();
                                }
                                if (i_wi_r_1 == 0) {
                                    int_res_c[(l_cb_offset_l_1 + (((p_cb_offset_l_1 + (((p_iteration_l_1)) / 1) * 1 + 0)) / 1) * (1 * 1) + i_wi_l_1)][(l_cb_offset_l_2 + (((p_cb_offset_l_2 + (((p_iteration_l_2)) / 1) * 20 + 0)) / 20) * (50 * 20) + i_wi_l_2)] = c_l_reduction_mem[i_wi_l_1][i_wi_r_1][i_wi_l_2];
                                }
                                __syncthreads();
                            }
                        }
                        p_cb_offset_l_2 += 20 * (1);
                    }
                    p_cb_offset_l_1 += 1 * (1);
                }
            }
            l_cb_offset_l_2 += (50 * 20) * (20 / 20);
        }
        l_cb_offset_l_1 += (1 * 1) * (1 / 1);
    }
}
