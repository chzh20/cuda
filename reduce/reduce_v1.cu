#include"cuda_runtime.h"
#include"device_launch_parameters.h"



/*

reduce_basic 存在以下问题：
1. 取模操作非常耗时
2. if(tid % (2 * stride) == 0) 导致了大量的control divergence，即线程分支非常多，导致了大量的warp divergence
3. 没有使用shared memory，导致了大量的global memory访问，global memory访问非常慢
*/

template<typename T, typename OP>
__global__ void reduce_basic(const T * input,T* output, int n, OP op)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n)
    {
        return;
    }
    T* data = input + blockIdx.x * blockDim.x;
    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if(tid % (2 * stride) == 0)
        {
            data[tid] = op(data[tid], data[tid + stride]);
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        output[blockIdx.x] = data[0];
    }

}





/*
reduce_v1 有以下优点：
1.使用shared memory，减少了global memory访问
2.通过 int index = 2 * stride * tid; 较少了取模操作和control divergence
reduce_v1每个warp中都有control divergence；而redece_v2中只有第一个warp有control divergence(当线程数小于32)

有以下缺点：
1. 仍然存在大量的warp divergence.
2. 有一半的线程没有工作，浪费了计算资源.
假设blocksize=256,则分配256/32=8个warp.只有前4个warp有工作，后4个warp没有工作.

*/

template<typename T, int BlockSize, typename OP>
__global__ void reduce_v1(const T * input,T* output, int n, OP op)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n)
    {
        return;
    }
    T* data = input + blockIdx.x * blockDim.x;
    __shared__  T smem[BlockSize];
    smem[tid] = data[tid];
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;
        if(index < blockDim.x ) // 有一半的线程没有工作
        {
            smem[index] = op(smem[index], smem[index + stride]);
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}


/*
序列访问元素，减少warp divergence

*/

template<typename T, int BlockSize, typename OP>
__global__ void reduce_v2(const T * input,T* output, int n, OP op)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n)
    {
        return;
    }
    T* data = input + blockIdx.x * blockDim.x;
    __shared__  T smem[BlockSize];
    smem[tid] = data[tid];
    __syncthreads();

    for(int stride =  blockDim.x/2; stride > 0; stride >>= 1)
    {
        if( tid < stride )
        {
            smem[index] = op(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}

