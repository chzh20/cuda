#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<cstdio>


constexpr int TILE_SIZE = 32;

template<typename T>
__global__ void sgemm_shared(int m, int n, int k, T alpha, const T*A, const T *B, T beta,T*C)
{
    // Shared memory for tiles of A and B
    __shared__ T shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ T shared_B[TILE_SIZE][TILE_SIZE];

    // Initialize the result for this thread
    float result = 0.0f;
 
    // Thread index
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    // Calculate the row and column index of the element
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Loop over tiles
    for(int i = 0; i < k; i += TILE_SIZE)
    {
        // Load tiles into shared memory
        if (row < m && (i + tx) < k)
            shared_A[ty][tx] = A[row * k + i + tx];
        else
            shared_A[ty][tx] = 0.0f;

        if (col < n && (i + ty) < k)
            shared_B[ty][tx] = B[(i + ty) * n + col];
        else
            shared_B[ty][tx] = 0.0f;

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Compute partial product
        for(int j = 0; j < TILE_SIZE; j++)
        {
            result += shared_A[ty][j] * shared_B[j][tx];
        }

        // Synchronize to make sure that computation is done before loading new tiles
        __syncthreads();
    }

    // Write the result to the output matrix
    if(row < m && col < n)
    {
        C[row * n + col] = alpha * result + beta * C[row * n + col];
    }
}

template<typename T>
__global__ void sgemm_shared2(int m, int n, int k, T alpha, const T*A, const T *B, T beta,T*C)
{
    // Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Tile dimensions
    const int BM = TILE_SIZE;
    const int BN = TILE_SIZE;
    const int BK = TILE_SIZE;

    // Shared memory for tiles of A and B
    __shared__ T shared_A[BM * BK];
    __shared__ T shared_B[BK * BN];

    // Advance pointers to the start of the block
    A += by * BM * k;  // advance A to point to the row of blocks, row = by * BM
    B += bx * BN;  // advance B to point to the right block, col = bx * BN
    C += by * BM * n + bx * BN; // advance C to point to the right block, row = by * BM, col = bx * BN
    
    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Initialize result
    T result = 0.0f;

    // Loop over tiles
    for(int i = 0; i < k; i += BK)
    {   
        // Load tiles into shared memory
        shared_A[ty * BK + tx] = A[ty * k + tx];
        shared_B[ty * BK + tx] = B[ty * n + tx];
        __syncthreads();
        
        // Advance pointers to the next tile
        A += BK;
        B += BK * n;

        // Compute partial product
        for(int j = 0; j < BK; j++)
        {
            result += shared_A[ty * BK + j] * shared_B[j * BN + tx];
        }
        __syncthreads();
    }

    // Write the result to the output matrix
    C[ty * n + tx] = alpha * result + beta * C[ty * n + tx];
}
