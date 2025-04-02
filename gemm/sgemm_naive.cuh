#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<cstdio>



/*
    C = alpha * A * B + beta * C which A is m x k, B is k x n, C is m x n
*/
template<typename T>
__global__ void sgemm_naive(int m, int n, int k, T alpha, const T*A, const T *B, T beta,T*C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        T result = 0;
        for (int i = 0; i < k; i++)
        {
            result += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = alpha * result + beta * C[row * n + col];
    }
}

template<typename T>
__global__ void sgemm_naive_tans(int m, int n, int k, T alpha, const T*A, const T *B, T beta,T*C)
{
    //int row = blockIdx.y * blockDim.y + threadIdx.y;
    //int col = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x < m &&  y < n)
    {
        T result = 0;
        for (int i = 0; i < k; i++)
        {   
            result += A[x * k + i] * B[i * n + y];
        }
        C[x * n + y] = alpha * result + beta * C[x * n + y];
    }
}


template<typename T>
__global__ void sgemm_coalescing(int m, int n, int k, T alpha, const T*A, const T *B, T beta,T*C)
{
    const int block_size = blockDim.x;
    int x = blockIdx.x * block_size + (threadIdx.x / block_size);
    int y = blockIdx.y * block_size + (threadIdx.x % block_size);

    if( x < m &&  y < n)
    {
        T result = 0;
        for (int i = 0; i < k; i++)
        {
            result += A[x * k + i] * B[i * n + y];
        }
        C[x * n + y] = alpha * result + beta * C[x * n + y];
    }
}