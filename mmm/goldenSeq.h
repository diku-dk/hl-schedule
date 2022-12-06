
/**
 * Computes matrix multiplication C = A*B
 * Semantically the matrix sizes are:
 *    A : [heightA][widthA]T
 *    B : [ widthA][widthB]T
 *    C : [heightA][widthB]T
 *  for some numeric type T.
 **/
template<class T>
void goldenSeq(T* A, T* B, T* C, int heightA, int widthB, int widthA) {
    for(int i=0; i<heightA; i++) {
        for(int j=0; j<widthB; j++) {
            T sum = 0;
            for(int k=0; k<widthA; k++) {
                sum += A[i*widthA +k] * B[k*widthB + j];
            }
            C[i*widthB + j] = sum;
        }
    }
}

/**
 * An LMAD L = off + { (n_1 : s_1), ... (n_k, s_k) }
 *   denotes the set of 1D points (memory locations):
 *   { off + i_1*s_1 + ... + i_k*s_k |
 *     0 <= i_1 < n_1, ..., 0 <= i_k < n_k }
 */

/**
 * HL-Schedule of kernel mmmSymBlkRegAllParKer (in kernels.cu.h)
 * We name the sizes shorter as:
 *   A : [M, Q]T
 *   B : [Q, N]T
 *   C : [M, N]T
 * Observation: We might need to map all variables to a memory space,
 *       and/or we might also need to declare the memory type in which
 *                a variable is primarily computed (reg, shmem, glbmem). 
 *
 *
 * splitable M = m * Tby, N = n * Tbx, Q = q * Ts
 *
 * @Grid g(bid.z < q, bid.y < m, bid.x < n):
 * W(C) = C[:,:].split(1).split(0).pushRed(q).reorder([1,3,2,4,0]).mapPar(0->g.z, 1->g.y, 2->g.x)
 * R(A) = A.split(1).split(0).reorder([1,2,0,3]).mapPar(0->g.z, 1->g.y)
 * R(B) = B.split(1).split(0).reorder([0,2,1,3]).mapPar(0->g.z, 1->g.x)
 *
 * //Produces LMADs:
 * // W(C) = 0 + { (q : 0)^{g.z}, (m : N*Tby)^{g.y}, (n : Tbx)^{g.x}, (Tby:N), (Tbx:1) }
 * // R(A) = 0 + { (q : Ts)^{g.z}, (m : Q*Tby)^{g.y}, (Tby : Q), (Ts : 1) }
 * // R(B) = 0 + { (q : N*Ts)^{g.z}, (n : Tbx)^{g.x}, (Ts : N), (Tbx : 1) }
 *
 * splitable Tby = Ty*Ry, Tbx = Tx*Rx, Ts = Tk*Rk
 *
 * @Block b(tid.y < Ty, tid.x < Tx):
 * W(C) = W(C).g[@,@,@].split(1).split(0).reorder(0,2,1,3).mapPar(0->b.y, 1->b.x)
 *    // [Ty][Tx][Ry][Rx]
 * 
 * R(A) = R(A).g[@,@].split(1).split(0).reorder(1,2,3,0).mapPar(1 -> b.y)
 *    // [Rk][Ty][Ry][Tk], with Ty mapped on block.y
 *
 * R(B) = R(B).g[@,@].split(1).split(0).reorder(1,0,2,3).mapPar(2 -> b.x)
 *    // [Rk][Tk][Tx][Rx], with Tx mapped on block.x
 *
 * Ash = glb2sh( R(A)[@] )
 *    // shared-mem buffer of size: [Ty][Ry][Tk]
 * Bsh = glb2sh( R(B)[@] )
 *    // shared-mem buffer of size: [Tk][Tx][Rx]
 *
 * @Thread t:
 * Areg = sh2reg( Ash[@,:,@] ) // [Ry]
 * Breg = sh2reg( Bsh[@,@,:] ) // [Rx]
 * Creg = compute( new reg( Ry, Rx ) ) // [Ry][Rx]
 * update( W(C).b[@,@], Creg )
 *
 * // Comments:
 * //   1. can we get rid of reorder, at least when non-ambiguous (?)
 * //   2. if a size, e.g., M is splitable in multiple ways, then
 * //      we need to qualify the spliting to make it unique.
 * //   3. don't like: same notation for set of indices and array slice (?)
 */
