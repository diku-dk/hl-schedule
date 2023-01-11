__kernel void batch_matmul(__global float const * const restrict a_buf, __global float const * const restrict b_buf, __global float * const restrict int_res_c) {
    size_t l_cb_offset_l_1, l_cb_offset_l_2, l_cb_offset_l_3, l_cb_offset_r_1, p_cb_offset_l_1, p_cb_offset_l_2, p_cb_offset_l_3, p_cb_offset_r_1;

    const size_t i_wg_l_1 = get_group_id(1);
    const size_t i_wi_l_1 = 0;
    const size_t i_wg_l_2 = get_group_id(0);
    const size_t i_wi_l_2 = 0;
    const size_t i_wg_l_3 = (get_group_id(2) / (1));
    const size_t i_wi_l_3 = 0;
    const size_t i_wg_r_1 = 0;
    const size_t i_wi_r_1 = 0;

    __private float cb_p_b[(1)][(2)][(1)];
    __private float res_p_c[1][((10 / 1) / (2)) + (((1 * (10 / 1)) % (1 * (2)) / 1) > 0) + (((1 * (10 / 1)) % (1 * (2)) % 1) > 0)][2][((1 / 1) / (1)) + (((1 * (1 / 1)) % (1 * (1)) / 1) > 0) + (((1 * (1 / 1)) % (1 * (1)) % 1) > 0)][1][((2 / 1) / (2)) + (((1 * (2 / 1)) % (1 * (2)) / 1) > 0) + (((1 * (2 / 1)) % (1 * (2)) % 1) > 0)][2];

    l_cb_offset_l_1 = i_wg_l_1 * 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t l_step_l_1 = 0; l_step_l_1 < ((16 / (16 * 1)) / (1 / 1)); ++l_step_l_1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        l_cb_offset_l_2 = i_wg_l_2 * 1;
        for (size_t l_step_l_2 = 0; l_step_l_2 < ((10 / (5 * 1)) / (2 / 1)); ++l_step_l_2) {
            barrier(CLK_LOCAL_MEM_FENCE);
            l_cb_offset_l_3 = i_wg_l_3 * 1;
            for (size_t l_step_l_3 = 0; l_step_l_3 < ((500 / (2 * 1)) / (10 / 1)); ++l_step_l_3) {
                barrier(CLK_LOCAL_MEM_FENCE);
                l_cb_offset_r_1 = i_wg_r_1 * 1;
                size_t l_step_r_1 = 0;
                p_cb_offset_l_1 = i_wi_l_1 * 1;
                for (size_t p_step_l_1 = 0; p_step_l_1 < ((1 / 1) / (1)); ++p_step_l_1) {
                    p_cb_offset_l_2 = i_wi_l_2 * 1;
                    for (size_t p_step_l_2 = 0; p_step_l_2 < ((2 / 1) / (2)); ++p_step_l_2) {
                        p_cb_offset_l_3 = i_wi_l_3 * 1;
                        for (size_t p_step_l_3 = 0; p_step_l_3 < ((10 / 1) / (2)); ++p_step_l_3) {
                            {
                                size_t p_step_r_1 = 0;
                                {
                                    for (size_t step = 0; step < ((1)) * ((1)) * ((2)) / (1); ++step) {
                                        const size_t flat_index = (0) + step * (1);
                                        const size_t p_dim_0_index_l_1 = flat_index / (((1)) * ((2)));
                                        const size_t p_dim_1_index_r_1 = (flat_index / (((2)))) % ((1));
                                        const size_t p_dim_2_index_l_3 = flat_index % ((2));
                                        cb_p_b[((p_dim_1_index_r_1))][((p_dim_2_index_l_3))][((p_dim_0_index_l_1))] = b_buf[(((l_step_l_1 * (1 / 1) + (((p_step_l_1 * (1) + (p_dim_0_index_l_1) / 1) * 1 + i_wi_l_1 * 1 + ((p_dim_0_index_l_1) % 1))) / 1) * (16 * 1) + i_wg_l_1 * 1 + ((((p_step_l_1 * (1) + (p_dim_0_index_l_1) / 1) * 1 + i_wi_l_1 * 1 + ((p_dim_0_index_l_1) % 1))) % 1))) * (64) * (500) + (((l_step_r_1 * (1 / 1) + (((p_step_r_1 * (1) + (p_dim_1_index_r_1) / 1) * 1 + i_wi_r_1 * 1 + ((p_dim_1_index_r_1) % 1))) / 1) * (1 * 1) + i_wg_r_1 * 1 + ((((p_step_r_1 * (1) + (p_dim_1_index_r_1) / 1) * 1 + i_wi_r_1 * 1 + ((p_dim_1_index_r_1) % 1))) % 1))) * (500) + (((l_step_l_3 * (10 / 1) + (((p_step_l_3 * (2) + (p_dim_2_index_l_3) / 1) * 1 + i_wi_l_3 * 1 + ((p_dim_2_index_l_3) % 1))) / 1) * (2 * 1) + i_wg_l_3 * 1 + ((((p_step_l_3 * (2) + (p_dim_2_index_l_3) / 1) * 1 + i_wi_l_3 * 1 + ((p_dim_2_index_l_3) % 1))) % 1)))];
                                    }
                                }
                            }
                            p_cb_offset_r_1 = i_wi_r_1 * 1;
                            size_t p_step_r_1 = 0;
                            #pragma unroll
                            for (size_t p_iteration_l_1 = 0; p_iteration_l_1 < (1); ++p_iteration_l_1) {
                                #pragma unroll
                                for (size_t p_iteration_l_2 = 0; p_iteration_l_2 < (2); ++p_iteration_l_2) {
                                    #pragma unroll
                                    for (size_t p_iteration_l_3 = 0; p_iteration_l_3 < (2); ++p_iteration_l_3) {
                                        size_t p_iteration_r_1 = 0;
                                        res_p_c[(0)][p_step_l_3][(p_iteration_l_3)][p_step_l_1][(p_iteration_l_1)][p_step_l_2][(p_iteration_l_2)] = a_buf[((l_cb_offset_l_1 + (((p_cb_offset_l_1 + (((p_iteration_l_1)) / 1) * 1 + 0)) / 1) * (16 * 1) + i_wi_l_1)) * (10) * (64) + ((l_cb_offset_l_2 + (((p_cb_offset_l_2 + (((p_iteration_l_2)) / 1) * 1 + 0)) / 1) * (5 * 1) + i_wi_l_2)) * (64) + ((l_cb_offset_r_1 + (((p_cb_offset_r_1 + (((p_iteration_r_1)) / 1) * 1 + 0)) / 1) * (1 * 1) + i_wi_r_1))] * cb_p_b[(((p_iteration_r_1)))][(((p_iteration_l_3)))][(((p_iteration_l_1)))];
                                    }
                                }
                            }
                            p_cb_offset_r_1 += 1 * (1);
                            p_cb_offset_l_3 += 1 * (2);
                        }
                        p_cb_offset_l_2 += 1 * (2);
                    }
                    p_cb_offset_l_1 += 1 * (1);
                }
                l_cb_offset_r_1 += (1 * 1) * (1 / 1);
                for (l_step_r_1 = 1; l_step_r_1 < ((64 / (1 * 1)) / (1 / 1)); ++l_step_r_1) {
                    p_cb_offset_l_1 = i_wi_l_1 * 1;
                    for (size_t p_step_l_1 = 0; p_step_l_1 < ((1 / 1) / (1)); ++p_step_l_1) {
                        p_cb_offset_l_2 = i_wi_l_2 * 1;
                        for (size_t p_step_l_2 = 0; p_step_l_2 < ((2 / 1) / (2)); ++p_step_l_2) {
                            p_cb_offset_l_3 = i_wi_l_3 * 1;
                            for (size_t p_step_l_3 = 0; p_step_l_3 < ((10 / 1) / (2)); ++p_step_l_3) {
                                {
                                    size_t p_step_r_1 = 0;
                                    {
                                        for (size_t step = 0; step < ((1)) * ((1)) * ((2)) / (1); ++step) {
                                            const size_t flat_index = (0) + step * (1);
                                            const size_t p_dim_0_index_l_1 = flat_index / (((1)) * ((2)));
                                            const size_t p_dim_1_index_r_1 = (flat_index / (((2)))) % ((1));
                                            const size_t p_dim_2_index_l_3 = flat_index % ((2));
                                            cb_p_b[((p_dim_1_index_r_1))][((p_dim_2_index_l_3))][((p_dim_0_index_l_1))] = b_buf[(((l_step_l_1 * (1 / 1) + (((p_step_l_1 * (1) + (p_dim_0_index_l_1) / 1) * 1 + i_wi_l_1 * 1 + ((p_dim_0_index_l_1) % 1))) / 1) * (16 * 1) + i_wg_l_1 * 1 + ((((p_step_l_1 * (1) + (p_dim_0_index_l_1) / 1) * 1 + i_wi_l_1 * 1 + ((p_dim_0_index_l_1) % 1))) % 1))) * (64) * (500) + (((l_step_r_1 * (1 / 1) + (((p_step_r_1 * (1) + (p_dim_1_index_r_1) / 1) * 1 + i_wi_r_1 * 1 + ((p_dim_1_index_r_1) % 1))) / 1) * (1 * 1) + i_wg_r_1 * 1 + ((((p_step_r_1 * (1) + (p_dim_1_index_r_1) / 1) * 1 + i_wi_r_1 * 1 + ((p_dim_1_index_r_1) % 1))) % 1))) * (500) + (((l_step_l_3 * (10 / 1) + (((p_step_l_3 * (2) + (p_dim_2_index_l_3) / 1) * 1 + i_wi_l_3 * 1 + ((p_dim_2_index_l_3) % 1))) / 1) * (2 * 1) + i_wg_l_3 * 1 + ((((p_step_l_3 * (2) + (p_dim_2_index_l_3) / 1) * 1 + i_wi_l_3 * 1 + ((p_dim_2_index_l_3) % 1))) % 1)))];
                                        }
                                    }
                                }
                                p_cb_offset_r_1 = i_wi_r_1 * 1;
                                size_t p_step_r_1 = 0;
                                #pragma unroll
                                for (size_t p_iteration_l_1 = 0; p_iteration_l_1 < (1); ++p_iteration_l_1) {
                                    for (size_t p_iteration_l_2 = 0; p_iteration_l_2 < (2); ++p_iteration_l_2) {
                                        for (size_t p_iteration_l_3 = 0; p_iteration_l_3 < (2); ++p_iteration_l_3) {
                                            for (size_t p_iteration_r_1 = 0; p_iteration_r_1 < (1); ++p_iteration_r_1) {
                                                res_p_c[(0)][p_step_l_3][(p_iteration_l_3)][p_step_l_1][(p_iteration_l_1)][p_step_l_2][(p_iteration_l_2)] += a_buf[((l_cb_offset_l_1 + (((p_cb_offset_l_1 + (((p_iteration_l_1)) / 1) * 1 + 0)) / 1) * (16 * 1) + i_wi_l_1)) * (10) * (64) + ((l_cb_offset_l_2 + (((p_cb_offset_l_2 + (((p_iteration_l_2)) / 1) * 1 + 0)) / 1) * (5 * 1) + i_wi_l_2)) * (64) + ((l_cb_offset_r_1 + (((p_cb_offset_r_1 + (((p_iteration_r_1)) / 1) * 1 + 0)) / 1) * (1 * 1) + i_wi_r_1))] * cb_p_b[(((p_iteration_r_1)))][(((p_iteration_l_3)))][(((p_iteration_l_1)))];
                                            }
                                        }
                                    }
                                }
                                p_cb_offset_r_1 += 1 * (1);
                                p_cb_offset_l_3 += 1 * (2);
                            }
                            p_cb_offset_l_2 += 1 * (2);
                        }
                        p_cb_offset_l_1 += 1 * (1);
                    }
                    l_cb_offset_r_1 += (1 * 1) * (1 / 1);
                }
                {
                    if (i_wi_r_1 == 0)
                    {
                        p_cb_offset_l_1 = i_wi_l_1 * 1;
                        for (size_t p_step_l_1 = 0; p_step_l_1 < ((1 / 1) / (1)); ++p_step_l_1) {
                            p_cb_offset_l_2 = i_wi_l_2 * 1;
                            for (size_t p_step_l_2 = 0; p_step_l_2 < ((2 / 1) / (2)); ++p_step_l_2) {
                                p_cb_offset_l_3 = i_wi_l_3 * 1;
                                for (size_t p_step_l_3 = 0; p_step_l_3 < ((10 / 1) / (2)); ++p_step_l_3) {
                                    for (size_t p_iteration_l_1 = 0; p_iteration_l_1 < (1); ++p_iteration_l_1) {
                                        for (size_t p_iteration_l_2 = 0; p_iteration_l_2 < (2); ++p_iteration_l_2) {
                                            for (size_t p_iteration_l_3 = 0; p_iteration_l_3 < (2); ++p_iteration_l_3) {
                                                int_res_c[((l_cb_offset_l_1 + (((p_cb_offset_l_1 + (((p_iteration_l_1)) / 1) * 1 + 0)) / 1) * (16 * 1) + i_wi_l_1)) * (((10 - 1)) - (0) + 1) * (((500 - 1)) - (0) + 1) + ((l_cb_offset_l_2 + (((p_cb_offset_l_2 + (((p_iteration_l_2)) / 1) * 1 + 0)) / 1) * (5 * 1) + i_wi_l_2)) * (((500 - 1)) - (0) + 1) + ((l_cb_offset_l_3 + (((p_cb_offset_l_3 + (((p_iteration_l_3)) / 1) * 1 + 0)) / 1) * (2 * 1) + i_wi_l_3))] = res_p_c[(0)][p_step_l_3][(p_iteration_l_3)][p_step_l_1][(p_iteration_l_1)][p_step_l_2][(p_iteration_l_2)];
                                            }
                                        }
                                    }
                                    p_cb_offset_l_3 += 1 * (2);
                                }
                                p_cb_offset_l_2 += 1 * (2);
                            }
                            p_cb_offset_l_1 += 1 * (1);
                        }
                    }
                }
                l_cb_offset_l_3 += (2 * 1) * (10 / 1);
            }
            l_cb_offset_l_2 += (5 * 1) * (2 / 1);
        }
        l_cb_offset_l_1 += (16 * 1) * (1 / 1);
    }
}
