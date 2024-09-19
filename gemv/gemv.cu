#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include <__clang_cuda_builtin_vars.h>
#include <climits>
#include <cstddef>
#include<iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include<vector>
#include<string>
#include<random>
#include<functional>
#include<tuple>
#include"cublas_v2.h"
#include<cuda_fp16.h>
template <typename T>
void check(T result, const char *function, const char *file, size_t line)
{
    if (result)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " function " << function
                  << " error code: " << cudaGetErrorName(result)
                  << " error string: " << cudaGetErrorString(result) << std::endl;
        // Optionally, you might want to reset the CUDA error state
        // cudaGetLastError(); // To reset the error state
        exit(EXIT_FAILURE); // EXIT_FAILURE is more standard than 1
    }
}

#define CUDACHECK(val) do { check((val), #val, __FILE__, __LINE__); } while (0)



template<typename T>
void gemvCPU(const T* mat,const T*vec,T*res,int m,int n)
{
    for(int i =0; i<m;++i)
    {
        res[i] = 0;
        for(int j =0; j<n;++j)
        {
            res[i] += mat[i*n+j]*vec[j];
        }
    }
}
template<typename T>
bool checkGroundTruth(const T* res1,const T* res2,int m)
{
    for(int i =0; i<m;++i)
    {
        if(fabs(res1[i]-res2[i])>1e-3)
        {
            std::cerr<<"Error at "<<i<<" "<<res1[i]<<" "<<res2[i]<<std::endl;
            return false;
        }
    }
    return true;
}


template<typename T>
struct Vec{
    static constexpr size_t size =4;
};
template<>
struct Vec<half2>{
    static constexpr size_t size = 8;
};


template<template<typename> typename ReductionOp,typename T>
__device__ __forceinline__ T warpReduce(T val)
{
    for(int offset = warpSize/2; offset>0;offset/=2)
    {
        val = ReductionOp<T>()(val,__shfl_down_sync(0xffffffff,val,offset));
    }
    return val;
}

template<template<typename> typename ReductionOp,typename T>
__device__ __forceinline__ T blockReduce(T val)
{
    static __shared__ T shared[64]; //warpsize 2048/32 = 64
    int tid = threadIdx.x;
    int lane = threadIdx.x%warpSize;
    int wid = threadIdx.x/warpSize;
    int  warp_nums = (blockDim.x+warpSize-1)/warpSize;
    val = warpReduce<ReductionOp,T>(val);
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();
    T wrap_val = (tid<warp_nums)?shared[tid]:T(0);
    return  warpReduce<ReductionOp,T>(wrap_val);
}


template<typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a,const T &b)
    {
        return a+b;
    }
};
template<>
struct SumOp<half>
{
    __device__ __forceinline__ half operator()(const half &a,const half &b)
    {
        return __hadd(a,b);
    }
};



//compute one element per block
//m blocks
template<size_t VECS_PER_THREAD,size_t VEC_SIZE>
__global__ void gemvKernel(float* mat,float* vec,float* dst,int m,int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x; // row index
    float thread_local_sum = 0.0f;
    for(int i =0; i< VECS_PER_THREAD;++i)
    {
        float4 * mat4 = reinterpret_cast<float4*>(&mat[bid*n+ tid*VEC_SIZE]);
        float4 * vec4 = reinterpret_cast<float4*>(&vec[tid*VEC_SIZE]);
        thread_local_sum += mat4[i].x*vec4[i].x+mat4[i].y*vec4[i].y+mat4[i].z*vec4[i].z+mat4[i].w*vec4[i].w;
    }
    float block_sum = blockReduce<SumOp,float>(thread_local_sum);
    if(tid == 0)
    {
        dst[bid] = block_sum;
    }
    __syncthreads();

}

template<size_t VECS_PER_THREAD,size_t VEC_SIZE>
__global__ void gemvKernel(half* mat,half* vec,half* dst,int m,int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    half thread_local_sum = 0.0f;
    for(int i = 0; i< VECS_PER_THREAD; ++i)
    {
       float4 * mat4 = reinterpret_cast<float4*>(&mat[bid*n+tid*VEC_SIZE]);
       float4 * vec4 = reinterpret_cast<float4*>(&vec[tid*VEC_SIZE]);
       half2*   vec_h1 =(half2*)&vec4[i].x;
       half2*   vec_h2 =(half2*)&vec4[i].y;
       half2*   vec_h3 = (half2*)&vec4[i].z;
       half2*   vec_h4 = (half2*)&vec4[i].w;

       half2*   mat_h1 = (half2*)&mat4[i].x;
       half2*   mat_h2 = (half2*)&mat4[i].y;
       half2*   mat_h3 = (half2*)&mat4[i].z;
       half2*   mat_h4 = (half2*)&mat4[i].w;
       half2 res1 = __hmul2(*mat_h1,*vec_h1);
       half2 res2 = __hmul2(*mat_h2,*vec_h2);
       half2 res3 =__hmul2(*mat_h3,*vec_h3);
       half2 res4 = __hmul2(*mat_h4,*vec_h4);
       half2 res = __hadd2(__hadd2(res1, res2), __hadd2(res3, res4));
       thread_local_sum = __hadd(res.x,res.y);
    }
    half block_sum = blockReduce<SumOp,half>(thread_local_sum);
    if(tid ==0)
    {
        dst[bid] = block_sum;
    }
    __syncthreads();
}



//VEC_SIZE表示每个向量的大小
//VECS_PER_THREAD表示每个线程处理的向量数
//THREAD_NUMS表示每个block的线程数
template<size_t VECS_PER_THREAD, size_t VEC_SIZE,size_t THREAD_NUMS>
struct DispatchLauncher
{
    template<typename T>
    static void launch(const T* mat,const T* vec,T* dst,int m,int n)
    {
        dim3 grid(m);
        dim3 block(THREAD_NUMS);
        float time=0.0f;
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);
        gemvKernel<T><<<grid,block>>>(mat,vec,dst,m,n);
        CUDACHECK(cudaGetLastError());
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time,start,stop);
        std::cout<<"Time: "<<time<<std::endl;
    }
};






template<typename T>
void gemv_kernel(T* mat,T*d_mat,T*vec,T*d_vec,T*dst,T*d_dst)
{
    constexpr size_t M = 256;
    constexpr size_t N = 2048;
    vec = (T*)malloc(N*sizeof(T));
    mat = (T*)malloc(M*N*sizeof(T));
    dst = (T*)malloc(M*sizeof(T));

    cudaMalloc(&d_mat,M*N*sizeof(T));
    cudaMalloc(&d_vec,N*sizeof(T));
    cudaMalloc(&d_dst,M*sizeof(T));

    for(int i=0;i<N;++i)
    {
        vec[i] = (T)(rand()%100)/100;
    }
    for(int i=0;i<M*N;++i)
    {
        mat[i] = (T)(rand()%100)/100;
    }

    cudaMemcpy(d_mat,mat,M*N*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec,vec,N*sizeof(T),cudaMemcpyHostToDevice);

    
    constexpr size_t THREAD_NUMS = 256;
    constexpr size_t VEC_SIZE = Vec<T>::size;
    constexpr size_t VECS_PER_THREAD = (N/THREAD_NUMS)/VEC_SIZE;

    DispatchLauncher<VECS_PER_THREAD,VEC_SIZE,THREAD_NUMS>::template launch(d_mat,d_vec,d_dst,M,N);

    cudaMemcpy(dst,d_dst,M*sizeof(T),cudaMemcpyDeviceToHost);

    T* dst_cpu = (T*)malloc(M*sizeof(T));
    gemvCPU(mat,vec,dst_cpu,M,N);
    if(!checkGroundTruth(dst,dst_cpu,M))
    {
        std::cerr<<"Error"<<std::endl;
    }
    else
    {
        std::cout<<"Success"<<std::endl;
    }
    free(vec);
    free(mat);
    free(dst);
    free(dst_cpu);
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_dst);


}
int main()
{
    return 0;
}