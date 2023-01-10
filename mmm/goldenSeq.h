
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
    //#hlsched@GPU SymBlkRegAllPar(heightA, widthB, widthA, A, B, C)
    for(int i=0; i<heightA; i++) {
        for(int j=0; j<widthB; j++) {
            T c = 0;
            for(int k=0; k<widthA; k++) {
                c += A[i*widthA +k] * B[k*widthB + j];
            }
            C[i*widthB + j] = c;
        }
    }
}
