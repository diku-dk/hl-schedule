__kernel void matmul(
        __global float const * const __restrict__ a_glb,
        __global float const * const __restrict__ b_glb,
        __global float * const __restrict__ c_glb) {
    __private float c_prv[1][8];
    __local float a_lcl[1][128];
    __local float b_lcl[128][8];

    const size_t wg_2 = get_group_id(0);

    for (size_t lcl_2 = 0; lcl_2 < 2; ++lcl_2) {
    #pragma unroll
    for (size_t prv_2 = 0; prv_2 < 4; ++prv_2) {
        c_prv[0][lcl_2 * 4 + prv_2] = 0.0f;
    }}

    for (size_t glb_3 = 0; glb_3 < 16; ++glb_3) {
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (size_t step = 0; step < 32; ++step) {
            ((__local float4*)a_lcl)[(step % 32)] =
            ((__global float4*)a_glb)[((glb_3 * 128) / 4) + (step % 32)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (size_t step = 0; step < 256; ++step)
        {
            ((__local float4*)b_lcl)[(step / 2 % 128) * 2 + (step % 2)] =
            ((__global float4*)b_glb)[((glb_3 * 128) + (step / 2 % 128)) * 250 + (((wg_2 * 8) / 4) + (step % 2))];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t lcl_3 = 0; lcl_3 < 64; ++lcl_3) {
        #pragma unroll
        for (size_t lcl_2 = 0; lcl_2 < 2; ++lcl_2) {
        #pragma unroll
        for (size_t prv_3 = 0; prv_3 < 2; ++prv_3) {
        #pragma unroll
        for (size_t prv_2 = 0; prv_2 < 4; ++prv_2) {
            c_prv[0][lcl_2 * 4 + prv_2] += a_lcl[0][lcl_3 * 2 + prv_3] * b_lcl[lcl_3 * 2 + prv_3][lcl_2 * 4 + prv_2];
        }}}}
    }

    for (size_t lcl_2 = 0; lcl_2 < 2; ++lcl_2) {
    for (size_t prv_2 = 0; prv_2 < 4; ++prv_2) {
        c_glb[wg_2 * 8 + lcl_2 * 4 + prv_2] = c_prv[0][lcl_2 * 4 + prv_2];
    }}
}

