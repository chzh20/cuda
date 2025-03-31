#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>
#include <cstdio>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;


// 增加单个线程的计算量，减少线程数
// 一个线程负责TM个元素
//dim3((BM*BN)/TM),  //block_size, each thread is responsible for TM elements
//dim3(CEIL_DIV(m,BM),CEIL_DIV(n,BN)), //grid_size
template <typename T>
__global__ void sgemm_1D_Blocktiling(int m, int n, int k, T alpha, const T *A,
                                     const T *B, T beta, T *C) {

  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // each blocktile is responsible for BM*BN elements
  const int totalResultsBlocktile = BM * BN;
  // a thread is reasponsible for TM  elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / TM;

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  A += by * BM * k;
  B += bx * BN;
  C += by * BM * n + bx * BN;

  // each warp is responsible for TM*32 elements in the blocktile
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // a thread is reasponsible for TM  elements
  T threadResult[TM] = {0.0f};
  // allocate shared memory for A and B
  __shared__ T shared_A[BM * BK];
  __shared__ T shared_B[BK * BN];

  assert(BM*BK == blockDim.x);
  assert(BK*BN == blockDim.x);
  //since bloksize is one dimension(BM*BK)/TM, we need use threadIdx.x to calculate the innerRow and innerCol
  const int innerColA = threadIdx.x % BK;
  const int innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN;
  const int innerRowB = threadIdx.x / BN;

  for (int bkIdx = 0; bkIdx < k; bkIdx += BK) {

    // load A and B into shared memory
    shared_A[innerRowA * BK + innerColA] = A[innerRowA * k + innerColA];
    shared_B[innerRowB * BN + innerColB] = B[innerRowB * n + innerColB];
    __syncthreads();


    A += BK;
    B += BK * n;
    // calculate per-thread results, traverse rows of B. or columns of A
    for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
      // each thread is responsible for a cloumn results in the blocktile
      // so we cache tempB in register
      T tempB = shared_B[dotIdx * BN + threadCol];
      for (int resIdx = 0; resIdx < TM; resIdx++) {
        threadResult[resIdx] +=
            shared_A[(threadRow * TM + resIdx) * BK + dotIdx] * tempB;
      }
    }
    __syncthreads();
  }
  // write the result to the output matrix
  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * n + threadCol] =
        alpha * threadResult[resIdx] +
        beta * C[(threadRow * TM + resIdx) * n + threadCol];
  }
}