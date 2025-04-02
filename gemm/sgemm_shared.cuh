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
    int threadCol = threadIdx.x; 
    int threadRow = threadIdx.y;

    // Calculate the row and column index of the element
    int row = blockIdx.y * TILE_SIZE + threadRow;
    int col = blockIdx.x * TILE_SIZE + threadCol;

    // Loop over tiles
    for(int i = 0; i < k; i += TILE_SIZE)
    {
        // Load tiles into shared memory
        if (row < m && (i + threadCol) < k)
            shared_A[threadRow][threadCol] = A[row * k + i + threadCol];
        else
            shared_A[threadRow][threadCol] = 0.0f;

        if (col < n && (i + threadRow) < k)
            shared_B[threadRow][threadCol] = B[(i + threadRow) * n + col];
        else
            shared_B[threadRow][threadCol] = 0.0f;

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Compute partial product
        for(int j = 0; j < TILE_SIZE; j++)
        {
            result += shared_A[threadRow][j] * shared_B[j][threadCol];
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
    const int cCol = blockIdx.x;
    const int cRow = blockIdx.y;

    // Tile dimensions
    const int BM = TILE_SIZE;
    const int BN = TILE_SIZE;
    const int BK = TILE_SIZE;

    // Shared memory for tiles of A and B
    __shared__ T shared_A[BM * BK];
    __shared__ T shared_B[BK * BN];

    // Advance pointers to the start of the block
    A += cRow * BM * k;  // advance A to point to the row of blocks, row = cRow * BM
    B += cCol * BN;  // advance B to point to the right block, col = cCol * BN
    C += cRow * BM * n + cCol * BN; // advance C to point to the right block, row = cRow * BM, col = cCol * BN
    
    // Thread index
    const int threadRow = threadIdx.x / TILE_SIZE ;
    const int threadCol = threadIdx.x % TILE_SIZE;

    // Initialize result
    T result = 0.0f;

    // Loop over tiles
    for(int i = 0; i < k; i += BK)
    {   
        // Load tiles into shared memory
        shared_A[threadRow * BK + threadCol] = A[threadRow * k + threadCol];
        shared_B[threadRow * BN + threadCol] = B[threadRow * n + threadCol];
        __syncthreads();
        
        // Advance pointers to the next tile
        A += BK;
        B += BK * n;

        // Compute partial product
        for(int j = 0; j < BK; j++)
        {
            result += shared_A[threadRow * BK + j] * shared_B[j * BN + threadCol];
        }
        __syncthreads();
    }

    // Write the result to the output matrix
    C[threadRow * n + threadCol] = alpha * result + beta * C[threadRow * n + threadCol];
}
