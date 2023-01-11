__kernel void convolution(
        __global float const * const __restrict__ images_glb,
        __global float const * const __restrict__ filter_glb,
        __global float * const __restrict__ out_glb) {
    __private float out_prv[1 * 1][2 * 1][7 * 2][1 * 1];
    __private float images_prv[1][(1) * 2 - (2 - 1) + (1) - 1][(2) * 2 - (2 - 1) + (1) - 1][1];
    __private float filter_prv[1][1][1][1];

    const size_t wg_2 = get_group_id(2) % (28);
    const size_t wg_3 = get_group_id(1);
    const size_t wi_2 = get_local_id(2) % (2);
    const size_t wi_4 = get_local_id(0);

    for (size_t glb_4 = 0; glb_4 < 2; ++glb_4) {
    for (size_t lcl_2 = 0; lcl_2 < 2; ++lcl_2) {
    for (size_t lcl_3 = 0; lcl_3 < 7; ++lcl_3) {
    for (size_t prv_3 = 0; prv_3 < 2; ++prv_3) {
        out_prv[0 * 1 + 0][lcl_2 * 1 + 0][lcl_3 * 2 + prv_3][0 * 1 + 0] = 0.0f;
    }}}}

    for (size_t glb_5 = 0; glb_5 < 3; ++glb_5) {
    for (size_t glb_6 = 0; glb_6 < 7; ++glb_6) {
    for (size_t glb_7 = 0; glb_7 < 7; ++glb_7) {
    for (size_t lcl_2 = 0; lcl_2 < 2; ++lcl_2) {
    for (size_t lcl_3 = 0; lcl_3 < 7; ++lcl_3) {
        for (size_t prv_3 = 0; prv_3 < 2; ++prv_3) {
            images_prv[0][(0) * 2 + (0)][(prv_3) * 2 + (0)][0] =
                    images_glb[(0 * 1 * 1 * 1 * 1 + 0 * 1 * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0) * ((2 * 112 + 7 - 1)) * ((2 * 112 + 7 - 1)) * (3) + ((0 * 28 * 2 * 2 * 1 + wg_2 * 2 * 2 * 1 + lcl_2 * 2 * 1 + wi_2 * 1 + 0) * 2 + (glb_6 * 1 * 1 * 1 * 1 + 0 * 1 * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0)) * ((2 * 112 + 7 - 1)) * (3) + ((0 * 8 * 7 * 1 * 2 + wg_3 * 7 * 1 * 2 + lcl_3 * 1 * 2 + 0 * 2 + prv_3) * 2 + (glb_7 * 1 * 1 * 1 * 1 + 0 * 1 * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0)) * (3) + (glb_5 * 1 * 1 * 1 * 1 + 0 * 1 * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0) ];
        }
        filter_prv[0][0][0][0] =
                filter_glb[(glb_4 * 1 * 1 * 32 * 1 + 0 * 1 * 32 * 1 + 0 * 32 * 1 + wi_4 * 1 + 0) * (7) * (7) * (3) + (glb_6 * 1 * 1 * 1 * 1 + 0 * 1 * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0) * (7) * (3) + (glb_7 * 1 * 1 * 1 * 1 + 0 * 1 * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0) * (3) + (glb_5 * 1 * 1 * 1 * 1 + 0 * 1 * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0) ];

        for (size_t prv_3 = 0; prv_3 < 2; ++prv_3) {
            out_prv[0 * 1 + 0][lcl_2 * 1 + 0][lcl_3 * 2 + prv_3][0 * 1 + 0] +=
                    images_prv[0][(0) * 2 + (0)][(prv_3) * 2 + (0)][0] *
                    filter_prv[0][0][0][0];
        }
    }}}}}

    for (size_t lcl_2 = 0; lcl_2 < 2; ++lcl_2) {
    for (size_t lcl_3 = 0; lcl_3 < 7; ++lcl_3) {
    for (size_t prv_3 = 0; prv_3 < 2; ++prv_3) {
        out_glb[(0 * 1 * 1 * 1 * 1 + 0 * 1 * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0) * (112) * (112) * (64) + (0 * 28 * 2 * 2 * 1 + wg_2 * 2 * 2 * 1 + lcl_2 * 2 * 1 + wi_2 * 1 + 0) * (112) * (64) + (0 * 8 * 7 * 1 * 2 + wg_3 * 7 * 1 * 2 + lcl_3 * 1 * 2 + 0 * 2 + prv_3) * (64) + (glb_4 * 1 * 1 * 32 * 1 + 0 * 1 * 32 * 1 + 0 * 32 * 1 + wi_4 * 1 + 0) ] =
                out_prv[0 * 1 + 0][lcl_2 * 1 + 0][lcl_3 * 2 + prv_3][0 * 1 + 0];
    }}}
}
