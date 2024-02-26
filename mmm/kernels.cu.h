#ifndef MULT_KERNELS
#define MULT_KERNELS

// widthA = heightB
template <class ElTp> 
__global__ void mmmNaiveKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (gidx >= widthB) || (gidy >= heightA) ) return;

  for(int k = 0; k < widthA; k ++) {
      accum += A[gidy*widthA + k] * B[k*widthB + gidx];
  }

  C[gidy*widthB + gidx] = accum;
}


template <class ElTp, int T> 
__global__ void mmmAsymBlkRegKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  __shared__ ElTp Ashreg[T][T];
  ElTp cs[T];

  unsigned int ii  = blockIdx.y * T;
  unsigned int jjj = blockIdx.x * T * T;
  unsigned int jj  = jjj + threadIdx.y * T;
  unsigned int j   = jj  + threadIdx.x;

  #pragma unroll
  for(int i=0; i<T; i++)
    cs[i] = 0.0;

  for(int kk = 0; kk < widthA; kk += T) {
      ElTp tmp = 0;
      if ((ii+threadIdx.y < heightA) && (kk+threadIdx.x < widthA)) {
        tmp = A[(ii+threadIdx.y)*widthA + kk+threadIdx.x];
      }
      Ashreg[threadIdx.y][threadIdx.x] = tmp;
      __syncthreads();

      for(int k = 0; k < T; k++) {
          ElTp b = 0;
          if ((k+kk < widthA) && (j < widthB)) {
            b = B[(k+kk)*widthB + j];
          }

          #pragma unroll 
          for(int i=0; i<T; i++) {
            cs[i] += Ashreg[i][k] * b;
          }
      }
      __syncthreads();
  }

  #pragma unroll
  for(int i=0; i<T; i++) {
    if( (ii+i < heightA) && (j < widthB) )
      C[(ii+i)*widthB + j] = cs[i];
  }
}

/************************************************/
/*** Block+Register Tile with different tiles ***/
/*** the parallel dimensions and the seq one  ***/
/************************************************/

template <class ElTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void mmmSymBlkRegInnSeqKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
#if 0
  __shared__ ElTp Aloc[Ty*Ry][Tk];
  __shared__ ElTp Bloc[Tk][Tx*Rx]; 
#else
  extern __shared__ uint64_t sh_mem_char[];
  ElTp* Aloc = (ElTp*) sh_mem_char;  // [Ty*Ry][Tk]
  ElTp* Bloc = (ElTp*) ( ((ElTp*)sh_mem_char) + Ty*Ry*Tk ); // [Tk][Tx*Rx]
#endif

  ElTp css[Ry][Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++)
    #pragma unroll
    for(int j=0; j<Rx; j++)
      css[i][j] = 0.0;

  for(int kk = 0; kk < widthA; kk += Tk) {

      // copy the slice of A: Ashreg = A[iii : iii + Ty*Ry , kk : kk+Tk]
      //   such that the accesses to A and Aloc are both coalesced!
      //for(int i = threadIdx.y; i < Ty*Ry; i+=Ty) {
      //    for(int k = threadIdx.x; k < Tk; k+=Tx) {
      for(int io = 0; io < Ry; io++) {
          for(int ko = 0; ko < (Tk + Tx - 1)/Tx; ko++) {
              int i = io*Ty + threadIdx.y;
              int k = ko*Tx + threadIdx.x;
              ElTp v = 0.0;
              if ( (iii+i < heightA) && (kk+k < widthA) )
                  v = A[(iii+i)*widthA + (kk+k)];
              //Aloc[i][k] = v;
              Aloc[i*Tk + k] = v;
          }
      }

      // copy the slice of B: Bshreg = B[kk : kk+Tk , jjj : jjj + Tx*Rx]
      //   such that the accesses to B and Bloc are both coalesced!
      //for(int k = threadIdx.y; k < Tk; k+=Ty) {
      //    for(int j = threadIdx.x; j < Tx*Rx; j+=Tx) {
      for(int ko = 0; ko < (Tk+Ty-1)/Ty; ko++) {
          for(int jo = 0; jo < Rx; jo++) {
              int k = ko*Ty + threadIdx.y;
              int j = jo*Tx + threadIdx.x;
              ElTp v = 0.0;
              if ( (jjj+j < widthB) && (kk+k < widthA) )
                  v = B[(kk+k)*widthB + (jjj + j)];
              //Bloc[k][j] = v;
              Bloc[k*(Tx*Rx) + j] = v;
          }
      }
      __syncthreads();

      for(int k = 0; k < Tk; k++) {
          // copy from local to register memory for A

          #pragma unroll
          for(int i=0; i<Ry; i++) {
            #pragma unroll
            for(int j=0; j<Rx; j++) {
                // unfortunately we need a safety condition here
                // or do we? because if i or j is out of range then
                // cs[i][j] is invalid anyways -- so everything looks safe!
                ////css[i][j] += as[i] * bs[j];
                css[i][j] += 
                  Aloc[ (threadIdx.y*Ry+i)*Tk + k] * //Aloc[threadIdx.y*Ry+i][k] * 
                  Bloc[k*(Tx*Rx) + (threadIdx.x*Rx+j)]; //Bloc[k][threadIdx.x*Rx+j] ;
            }
          }
      }
      __syncthreads();
  }

  const unsigned int indy = iii + threadIdx.y * Ry;

#if 0
  const unsigned int indx = jjj + threadIdx.x * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++) {
    #pragma unroll
    for(int j=0; j<Rx; j++) {
      if( (indy+i < heightA) && (indx+j < widthB) )
        C[(indy+i)*widthB + (indx+j)] = css[i][j];
    }
  }
#else
  #pragma unroll
  for(int i=0; i<Ry; i++) {
    for(int j=0; j<Rx; j++) {
      //Bloc[threadIdx.y][threadIdx.x*Rx+j] = css[i][j];
      Bloc[threadIdx.y*(Tx*Rx) + threadIdx.x*Rx+j] = css[i][j];
    }
    __syncthreads();
    for(int j=0; j<Rx; j++) {
      const unsigned int indxx = j*Tx + threadIdx.x;
      if( (indy+i < heightA) && (indxx+jjj < widthB) )
        C[(indy+i)*widthB + (indxx + jjj)] = Bloc[threadIdx.y*(Tx*Rx) + indxx]; //Bloc[threadIdx.y][indxx];
    }
    __syncthreads();
  }
#endif
}


template <class ElTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void mmmSymBlkRegInnSeqKerReg(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  __shared__ ElTp Aloc[Ty*Ry][Tk];
  __shared__ ElTp Bloc[Tk][Tx*Rx]; 
  ElTp css[Ry][Rx];
  ElTp as[Ry];
  ElTp bs[Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++)
    #pragma unroll
    for(int j=0; j<Rx; j++)
      css[i][j] = 0.0;

  for(int kk = 0; kk < widthA; kk += Tk) {

      // copy the slice of A: Ashreg = A[iii : iii + Ty*Ry , kk : kk+Tk]
      //   such that the accesses to A and Aloc are both coalesced!
      //for(int i = threadIdx.y; i < Ty*Ry; i+=Ty) {
      //    for(int k = threadIdx.x; k < Tk; k+=Tx) {
      for(int io = 0; io < Ry; io++) {
          for(int ko = 0; ko < (Tk + Tx - 1)/Tx; ko++) {
              int i = io*Ty + threadIdx.y;
              int k = ko*Tx + threadIdx.x;
              ElTp v = 0.0;
              if ( (iii+i < heightA) && (kk+k < widthA) )
                  v = A[(iii+i)*widthA + (kk+k)];
              Aloc[i][k] = v;
          }
      }

      // copy the slice of B: Bshreg = B[kk : kk+Tk , jjj : jjj + Tx*Rx]
      //   such that the accesses to B and Bloc are both coalesced!
      //for(int k = threadIdx.y; k < Tk; k+=Ty) {
      //    for(int j = threadIdx.x; j < Tx*Rx; j+=Tx) {
      for(int ko = 0; ko < (Tk+Ty-1)/Ty; ko++) {
          for(int jo = 0; jo < Rx; jo++) {
              int k = ko*Ty + threadIdx.y;
              int j = jo*Tx + threadIdx.x;
              ElTp v = 0.0;
              if ( (jjj+j < widthB) && (kk+k < widthA) )
                  v = B[(kk+k)*widthB + (jjj + j)];
              Bloc[k][j] = v;
          }
      }
      __syncthreads();

      for(int k = 0; k < Tk; k++) {
          // copy from local to register memory for A
          #pragma unroll
          for(int i = 0; i < Ry; i++) {
              as[i] = Aloc[threadIdx.y*Ry+i][k];
          }
          // copy from local to register memory for B
          #pragma unroll
          for(int j = 0; j < Rx; j++) {
              bs[j] = Bloc[k][threadIdx.x*Rx+j];
          }

          #pragma unroll
          for(int i=0; i<Ry; i++) {
            #pragma unroll
            for(int j=0; j<Rx; j++) {
                // unfortunately we need a safety condition here
                // or do we? because if i or j is out of range then
                // cs[i][j] is invalid anyways -- so everything looks safe!
                css[i][j] += as[i] * bs[j];
            }
          }
      }
      __syncthreads();
  }

  unsigned int indy = iii + threadIdx.y * Ry;
  unsigned int indx = jjj + threadIdx.x * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++) {
    #pragma unroll
    for(int j=0; j<Rx; j++) {
      if( (indy+i < heightA) && (indx+j < widthB) )
        C[(indy+i)*widthB + (indx+j)] = css[i][j];
    }
  }
}


/************************************************/
/*** All Dims Parallelized, including Redomap ***/
/************************************************/

template <class ElTp, int Ty, int Ry, int Tx, int Rx, int Tk, int Rk>
__global__ void mmmSymBlkRegAllParKer( 
                          ElTp* A, ElTp* B, ElTp* C, 
                          int heightA, int widthB, int widthA
) {
  __shared__ ElTp Aloc[Ty*Ry][Tk];
  __shared__ ElTp Bloc[Tk][Tx*Rx]; 
  ElTp css[Ry][Rx];
  ElTp as[Ry];
  ElTp bs[Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++)
    #pragma unroll
    for(int j=0; j<Rx; j++)
      css[i][j] = 0.0;

  unsigned int kkk = blockIdx.z * Tk * Rk;
  for(int kk = kkk; kk < min(widthA, kkk+Tk*Rk); kk += Tk) {

      // copy the slice of A: Ashreg = A[iii : iii + Ty*Ry , kk : kk+Tk]
      //   such that the accesses to A and Aloc are both coalesced!
      for(int i = threadIdx.y; i < Ty*Ry; i+=Ty) {
          for(int k = threadIdx.x; k < Tk; k+=Tx) {
              ElTp v = 0.0;
              if ( (iii+i < heightA) && (kk+k < widthA) )
                  v = A[(iii+i)*widthA + (kk+k)];
              Aloc[i][k] = v;
          }
      }

      // copy the slice of B: Bshreg = B[kk : kk+Tk , jjj : jjj + Tx*Rx]
      //   such that the accesses to B and Bloc are both coalesced!
      for(int k = threadIdx.y; k < Tk; k+=Ty) {
          for(int j = threadIdx.x; j < Tx*Rx; j+=Tx) {
              ElTp v = 0.0;
              if ( (jjj+j < widthB) && (kk+k < widthA) )
                  v = B[(kk+k)*widthB + (jjj + j)];
              Bloc[k][j] = v;
          }
      }
      __syncthreads();

      for(int k = 0; k < Tk; k++) {
          // copy from local to register memory for A
          #pragma unroll
          for(int i = 0; i < Ry; i++) {
              as[i] = Aloc[threadIdx.y*Ry+i][k];
          }
          // copy from local to register memory for B
          #pragma unroll
          for(int j = 0; j < Rx; j++) {
              bs[j] = Bloc[k][threadIdx.x*Rx+j];
          }

          #pragma unroll
          for(int i=0; i<Ry; i++) {
            #pragma unroll
            for(int j=0; j<Rx; j++) {
                // unfortunately we need a safety condition here
                // or do we? because if i or j is out of range then
                // cs[i][j] is invalid anyways -- so everything looks safe!
                css[i][j] += as[i] * bs[j];
            }
          }
      }
      __syncthreads();
  }

  //unsigned int indz = blockIdx.z * heightA * widthB;
  unsigned int indz = 0;
  unsigned int indy = iii + threadIdx.y * Ry;
  unsigned int indx = jjj + threadIdx.x * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++) {
    #pragma unroll
    for(int j=0; j<Rx; j++) {
      if( (indy+i < heightA) && (indx+j < widthB) )
        //C[indz + (indy+i)*widthB + (indx+j)] = css[i][j];
        atomicAdd( &C[indz + (indy+i)*widthB + (indx+j)], css[i][j]);
    }
  }
}

#endif
