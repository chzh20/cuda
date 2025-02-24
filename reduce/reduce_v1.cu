#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>


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


reduce1的最大问题是bank冲突。我们把目光聚焦在这个for循环中。并且只聚焦在0号warp。
在第一次迭代中，0号线程需要去load shared memory的0号地址以及1号地址的数，然后写回到0号地址。
而此时，这个warp中的16号线程，需要去load shared memory中的32号地址和33号地址。
可以发现，0号地址跟32号地址产生了2路的bank冲突。
在第2次迭代中，0号线程需要去load shared memory中的0号地址和2号地址。
这个warp中的8号线程需要load shared memory中的32号地址以及34号地址，
16号线程需要load shared memory中的64号地址和68号地址，24号线程需要load shared memory中的96号地址和100号地址。
又因为0、32、64、96号地址对应着同一个bank，所以此时产生了4路的bank冲突。现在，可以继续算下去，
8路bank冲突，16路bank冲突。由于bank冲突，所以reduce1性能受限。下图说明了在load第一个数据时所产生的bank冲突。
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

reduce_v2 有以下优点：
1. 通过调整交错访问，反转了元素的步幅：步幅从线程块大小的一半开始，然后在每次迭代中减半
每个线程在每轮中添加两个由当前步幅分隔的元素，以生成部分和。
2.减少了warp divergence：假设blocksize=256,则分配256/32=8个warp.
在第一轮中,warp0和warp1有工作，warp2和warp3没有工作
在第二轮中，warp0有工作，warp1没有工作，warp2和warp3有工作
在第三轮中，warp0和warp1没有工作，warp2有工作，warp3没有工作
2. 通过调整交错访问，减少了global memory访问

缺点：
1. stride =  blockDim.x/2： 有一半的线程没有工作，浪费了计算资源.

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


/* 

reduce_v3 有以下优点：
1.一次处理2个block的数据. 

*/

template <typename T, int BlockSize, typename OP>
__global__ void reduce_v3(const T *input, T *output, int n, OP op) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x * (blockDim.x * 2);
  T *data = input + blockIdx.x * (blockDim.x * 2);
  __shared__ T smem[BlockSize];
  if(tid + blockDim.x < n) {
    smem[tid] = op(data[tid], data[tid + blockDim.x]);
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[index] = op(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}
/*
reduce_v4 有以下优点：
当stride<=32时，只有第一个warp在工作
warp内，指令是SIMD同步, 在单个 warp 内，线程的执行是 SIMD 同步的，这意味着同一 warp 内的线程以相同的步调执行相同的指令。
因此，在 warp 内进行归约时，不需要额外的同步.
**Volatile 关键字**：`volatile` 关键字确保编译器不会对共享内存的访问进行优化，保证每次都从内存中读取最新的值。
因此可以把最后6次迭代展开
*/


template <typename T, int BlockSize, typename OP>
__global__ void reduce_v4(const T *input, T *output, int n, OP op) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x * (blockDim.x * 2);
  T *data = input + blockIdx.x * (blockDim.x * 2);
  __shared__ T smem[BlockSize];
  if(tid + blockDim.x < n) {
    smem[tid] = op(data[tid], data[tid + blockDim.x]);
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      smem[index] = op(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  //unroll the last 6 iterations
  if (tid < 32) {
    volatile T *vsmem = smem;
    vsmem[tid] = op(vsmem[tid], vsmem[tid + 32]);
    vsmem[tid] = op(vsmem[tid], vsmem[tid + 16]);
    vsmem[tid] = op(vsmem[tid], vsmem[tid + 8]);
    vsmem[tid] = op(vsmem[tid], vsmem[tid + 4]);
    vsmem[tid] = op(vsmem[tid], vsmem[tid + 2]);
    vsmem[tid] = op(vsmem[tid], vsmem[tid + 1]);
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}



// Simple functor for summation
struct AddOp {
    __host__ __device__ float operator()(float a, float b) const {
        return a + b;
    }
};

// Utility to check CUDA errors
static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " : " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void benchmarkKernels(int n, int blockSize) {
    // Host memory
    std::vector<float> hIn(n), hOut( (n + blockSize - 1) / blockSize );
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for(int i = 0; i < n; i++)
        hIn[i] = dist(gen);

    // Device memory
    float *dIn = nullptr, *dOut = nullptr;
    checkCuda(cudaMalloc(&dIn, n * sizeof(float)), "Failed to allocate dIn");
    checkCuda(cudaMalloc(&dOut, hOut.size() * sizeof(float)), "Failed to allocate dOut");
    checkCuda(cudaMemcpy(dIn, hIn.data(), n * sizeof(float), cudaMemcpyHostToDevice),
              "Failed to copy to dIn");

    // Helper lambda for timing
    auto timeKernel = [&](auto kernelLaunch, const char* label) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        kernelLaunch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << label << " took " << ms << " ms" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    // Grid size
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch each kernel (names assumed from existing code)
    timeKernel([&](){
        reduce_basic<float, AddOp><<<gridSize, blockSize>>>(dIn, dOut, n, AddOp());
        cudaDeviceSynchronize();
    }, "reduce_basic");

    timeKernel([&](){
        reduce_v1<float, 256, AddOp><<<gridSize, 256>>>(dIn, dOut, n, AddOp());
        cudaDeviceSynchronize();
    }, "reduce_v1");

    timeKernel([&](){
        reduce_v2<float, 256, AddOp><<<gridSize, 256>>>(dIn, dOut, n, AddOp());
        cudaDeviceSynchronize();
    }, "reduce_v2");

    timeKernel([&](){
        reduce_v3<float, 256, AddOp><<<gridSize, 256>>>(dIn, dOut, n, AddOp());
        cudaDeviceSynchronize();
    }, "reduce_v3");

    timeKernel([&](){
        reduce_v4<float, 256, AddOp><<<gridSize, 256>>>(dIn, dOut, n, AddOp());
        cudaDeviceSynchronize();
    }, "reduce_v4");

    // Cleanup
    cudaFree(dIn);
    cudaFree(dOut);
}

int main() {
    int n = 1 << 20;
    int blockSize = 256;
    benchmarkKernels(n, blockSize);
    return 0;
}