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
    //#hlsched@GPU spmvInnParSched(num_rows, rows, mat_inds, res)
    for(int i=0; i<num_rows; i++) {
        //ElTp acc = 0;
        res[i] = 0;
        const int beg_row = rows[i];
        const int end_row = rows[i+1];
        for(int j=beg_row; j<end_row; j++) {
            const int col_ind = mat_inds[j];
            res[i] += mat_vals[j] * vec[col_ind];
        }
        //res[i] = acc;
    }
}

#endif
