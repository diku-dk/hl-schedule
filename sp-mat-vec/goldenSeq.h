#ifndef GOLDEN
#define GOLDEN

/**
 * Sparse Matrix Vector Multiplication
 * Input:
 *    rows: the exclusive-scanned shape of the matrix,
 *          rows[i] denotes the start index of row i
 *    mat_inds: stores the column indices of the non-zeros
 *    mat_vals: the flat non-zero data of the matrix
 *    vec:      a vector
 *    num_rows: the number of rows of the matrix.
 * Result:
 *    res: the resulted vector (of length `num_rows`)
 * ElTp is some numeric type (single/double precision float).
 * 
 **/
template<class ElTp>
void goldenSeq  ( uint64_t*  rows
                , uint32_t*  mat_inds
                , ElTp*      mat_vals
                , ElTp*      vec
                , ElTp*      res
                , uint32_t   num_rows
) {
    for(int i=0; i<num_rows; i++) {
        ElTp acc = 0;
        const int beg_row = rows[i];
        const int end_row = rows[i+1];
        for(int j=beg_row; j<end_row; j++) {
            const int col_ind = mat_inds[j];
            acc += mat_vals[j] * vec[col_ind];
        }
        res[i] = acc;
    }
}

/**
 * An LMAD L = off + { (n_1 : s_1), ... (n_k, s_k) }
 *   denotes the set of 1D points (memory locations):
 *   { off + i_1*s_1 + ... + i_k*s_k |
 *     0 <= i_1 < n_1, ..., 0 <= i_k < n_k }
 */

/**
 * HL-Schedule of kernel "spmvInnParKer" (in kernels.cu.h)
 *
 * @Grid G(bid.x < num_rows):
 *
 * R(mat_inds) = ParU( G.x, mat_inds[ rows[bid.x] : rows[bid.x+1] ] )
 *
 * splitable R(mat_inds)[bid.x].length = q * b
 *
 * W(res) = res[:num_rows].mapPar(0 -> G.x)
 * 
 * @Block B(tid.x < b):
 *
 * res_sh = new shmem(b)
 * R(mat_inds) = R(mat_inds).G[bid.x].split(0) // [q][b]
 *
 * @Thread T:
 *
 * R(mat_inds) = R(mat_inds).B[:,tid.x]
 * res_reg = compute( new reg( 1 ) ) // [1]
 * res_sh = reg2sh(res_reg);
 * 
 * @Block B(tid.x < b):
 * W(res) = W(res).G[bid.x].pushRed(b)
 * W(res) = sh2gb(res_sh)
 *
 */

#endif
