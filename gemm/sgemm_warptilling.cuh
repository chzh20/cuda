
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>
#include <cstdio>

constexpr int NUM_THREADS = 128;
/*
BM: size of the block in the row dimension 


*/

template <typename T>
__global__ void sgemm_Warptiling(int m, int n, int k, T alpha, const T *A,
                                     const T *B, T beta, T *C)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 4;
    constexpr int WARPSIZE = 32;
    constexpr int WM = 64;
    constexpr int WN = 64;

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int warpIdx = threadIdx.x / WARPSIZE;
    const int warpRow = warpIdx / (BN/WN);
    const int warpCol = warpIdx % (BN/WN);
    //size of the warp subtile
    constexpr int WNITER = 2;
    constexpr int WMITER = (WM*WN)/(WARPSIZE*WNITER*TM*TN);
    constexpr int WSUBM = WM / WNITER;
    constexpr int WSUBN = WN / WNITER;

    // size of thread in the warp
    const int threadIdxInWarp = threadIdx.x % WARPSIZE;
    const int threadRowInWarp = threadIdxInWarp/(WSUBN/TN);
    const int threadColInWarp = threadIdxInWarp%(WSUBN/TN);

    __shared__ T shared_A[BM*BK];
    __shared__ T shared_B[BK*BN];


    A += cRow * BM * k;
    B += cCol * BN;
    C += (cRow*BM+ warpRow*WM)*n + cCol*BN + warpCol*WN;

    const int innerRowA = threadIdx.x / (BK/4);
    const int innerColA = threadIdx.x % (BK/4);
    const int strideA = (NUM_THREADS*4)/(BK);

    const int innerRowB = threadIdx.x /(BN/4);
    const int innerColB = threadIdx.x %(BN/4);
    const int strideB = (NUM_THREADS*4)/(BN);

    float threadResult[TM*TN*WNITER*WMITER] = {0.0f};
    float regM[TM*WMITER]={0.0f};
    float regN[TN*WNITER]={0.0f};

    for(int bkIdx = 0; bkIdx < k; bkIdx += BK)
    {
        // load data from global memory to shared memory
        for(int offset = 0; offset + strideA <= BM; offset += strideA)
        {
            const float4 temp = reinterpret_cast<const float4*>(&A[(innerRowA + offset) * k + innerColA * 4])[0];
            // 转置存储到shared_A
            //BM*BK --> BK*BM
            shared_A[(innerColA*4 + 0)*BM + (innerRowA + offset)] = temp.x;
            shared_A[(innerColA*4 + 1)*BM + (innerRowA + offset)] = temp.y;
            shared_A[(innerColA*4 + 2)*BM + (innerRowA + offset)] = temp.z;
            shared_A[(innerColA*4 + 3)*BM + (innerRowA + offset)] = temp.w;
        }

        for(int offset = 0; offset + strideB <= BK; offset += strideB)
        {
            const float4 temp = reinterpret_cast<const float4*>(&B[(innerRowB + offset) * n + innerColB * 4])[0];
            //float4 temp = const_cast<float4&>(tempB[0]);
            //float4 temp = reinterpret_cast<float4*>(&B[(innerRowB + offset)*n + innerColB*4])[0];
            reinterpret_cast<float4*>(&shared_B[(innerRowB + offset)*BN + innerColB*4])[0] = temp;
        }

        __syncthreads();
       
       for(int dotIdx =0; dotIdx <BK; dotIdx++)
       {

           // load data from shared memory to register
            for (int wSubRowIdx = 0; wSubRowIdx< WMITER; ++wSubRowIdx)
            {
                for (int i = 0; i < TM; ++i)
                {
                    regM[wSubRowIdx*TM + i] = shared_A[(dotIdx*BM)+ warpRow*WM + wSubRowIdx*WSUBM + threadRowInWarp*TM+ i];
                }
            }

            for(int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
            {
                for(int i = 0; i < TN; ++i)
                {
                    regN[wSubColIdx*TN + i] = shared_B[(dotIdx*BN) + warpCol*WN + wSubColIdx*WSUBN + threadColInWarp*TN +i];
                }
            }

            for(int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
            {
                for(int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                {
                    for(int i = 0; i < TM; ++i)
                    {
                        for(int j = 0; j < TN; ++j)
                        {
                            threadResult[(wSubRowIdx*TM + i)*WNITER*TN + wSubColIdx*TN + j] += regM[wSubRowIdx*TM + i] * regN[wSubColIdx*TN + j];
                        }
                    }
                }
            }

       }

        A += BK;
        B += BK * n;
        __syncthreads();
    }

    for(int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
    {
        for(int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
        {
            float* C_temp = C+ (wSubRowIdx*WSUBM)*n + wSubColIdx*WSUBN;
            for(int i = 0; i < TM; ++i)
            {
                for(int j = 0; j < TN; j+=4)
                {
                    float4 temp = reinterpret_cast<float4*>(&threadResult[(wSubRowIdx*TM + i)*WNITER*TN + wSubColIdx*TN + j])[0]; 
                    float4 tempC = reinterpret_cast<float4*>(&C_temp[(threadRowInWarp*TM+i)*n + threadColInWarp*TN+j])[0];
                    tempC.x = alpha * temp.x + beta * tempC.x;
                    tempC.y = alpha * temp.y + beta * tempC.y;
                    tempC.z = alpha * temp.z + beta * tempC.z;
                    tempC.w = alpha * temp.w + beta * tempC.w;
                    reinterpret_cast<float4*>(&C_temp[(threadRowInWarp*TM+i)*n + threadColInWarp*TN+j])[0]= tempC;
                }
            }
        }
    }
    
}