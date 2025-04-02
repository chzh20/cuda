#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>
#include <cstdio>
#if defined(_WIN64) || defined(_WIN32) 
#define uint unsigned int
#endif

template <typename T>
__global__ void sgemm_2D_Blocktiling(int m, int n, int k, T alpha, const T *A,
                                     const T *B, T beta, T *C) {
                            
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int totalResultsBlocktile = BM * BN;
    // a thread is reasponsible for TM*TN elements in the blocktile
    const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    assert(numThreadsBlocktile == blockDim.x);

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    //(BN/TN) is the number of threads in a column
    const int threadRow   = threadIdx.x /(BN/TN);
    const int threadCol   = threadIdx.x %(BN/TN);

    __shared__ T shared_A[BM * BK];
    __shared__ T shared_B[BK * BN];

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;

    //there are 128*8 elements in As; while only 256 threads
	//the shape of block in A is (32,8),while can not cover all elements,
	//so we need iterate along the row of A with strideA.
     //calculate the number of rows of Shared_A that are being loaded in one iteration
    const int strideA = numThreadsBlocktile/BK;

    
    //there are 8 * 128 elements in Bs; while only 256 threads
	//the shape of thread block in B is (2,128),while can not cover all elements,
	//so we need iterate along the row of B with strideB.
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    //calculate the number of rows of Shared_B that are being loaded in one iteration
    const int strideB = numThreadsBlocktile/BN;

    float threadResult[TM*TN] = {0.0f};
    float regM[TM]={0.0f};
    float regN[TN]={0.0f};

    
    for(uint bkIdx = 0;bkIdx <k; bkIdx += BK)
    {
        for(uint loadOffset = 0;loadOffset <BM; loadOffset += strideA)
        {
            
           shared_A[(innerRowA + loadOffset)*BK + innerColA] = A[(innerRowA + loadOffset)*k + innerColA]; 
        }
        for(uint loadOffset =0; loadOffset <BK; loadOffset += strideB)
        {
            shared_B[(innerRowB + loadOffset)*BN + innerColB] = B[(innerRowB + loadOffset)*n + innerColB];
        }

        __syncthreads();
        A += BK;
        B += BK * n;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            for(uint i = 0; i<TM; ++i)
            {
                // load TM rows of shared_A into register
                regM[i] = shared_A[(threadRow * TM + i)*BK + dotIdx];
            }
            for(uint i = 0; i<TN; ++i)
            {
                // load TN columns of shared_B into register
                regN[i] = shared_B[dotIdx * BN + threadCol * TN + i];
            }

            for(uint i = 0; i<TM; ++i)
            {
                for(uint j = 0; j<TN; ++j)
                {   // calculate the dot product of regM and regN
                    threadResult[i*TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }
    for(int i =0; i<TM; ++i)
    {
        for(int j =0; j<TN; ++j)
        {

            //now we have the result of TM*TN elements in the blocktile
            //we need to write it back to the global memory
            C[(threadRow * TM + i)*n + threadCol * TN + j] = alpha * threadResult[i*TN + j] + beta * C[(threadRow * TM + i)*n + threadCol * TN + j];
        }
    }
}
