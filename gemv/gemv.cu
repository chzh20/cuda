#include"cuda_runtime.h"
#include"device_launch_parameters.h"
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
#include <iomanip>
#include <iostream>
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

template <typename T>
void gemvCPU(const T *mat, const T *vec, T *res, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        res[i] = 0.0f;
        for (int j = 0; j < n; ++j)
        {
            res[i] += mat[i * n + j] * vec[j];
        }
    }
}
template <typename T>
bool checkGroundTruth(const T *res1, const T *res2, int m)
{
    for (int i = 0; i < m; ++i)
    {
        if (fabs(res1[i] - res2[i]) > 1e-3)
        {
            std::cerr << "Error at " << i << " " <<std::setw(5)<<std::fixed<< res1[i] << " " << res2[i] << std::endl;
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

template <template <typename> typename ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val = ReductionOp<T>()(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <template <typename> typename ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val)
{
    static __shared__ T shared[64]; // warpsize 2048/32 = 64
    int tid = threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    int warp_nums = (blockDim.x + warpSize - 1) / warpSize;
    val = warpReduce<ReductionOp, T>(val);
    if (lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();
    T wrap_val = (tid < warp_nums) ? shared[tid] : T(0);
    return warpReduce<ReductionOp, T>(wrap_val);
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
template <size_t VECS_PER_THREAD, size_t VEC_SIZE>
__global__ void gemvKernel(float *mat, float *vec, float *dst, int m, int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x; // row index
    float thread_local_sum = 0.0f;
    for (int i = 0; i < VECS_PER_THREAD; ++i)
    {
        int idx = blockDim.x + tid;
        if(idx<n)
        {
            float4 mat4 = reinterpret_cast<float4*>(mat)[bid * (n / VEC_SIZE) + idx]; // 1 * float4
            float4 vec4 = reinterpret_cast<float4*>(vec)[idx]; // after reinterpret_cast,vec[0]: v0,v1,v2,v3;vec[1]:v4,v5,v6,v7
            thread_local_sum += mat4.x * vec4.x;
            thread_local_sum += mat4.y * vec4.y;
            thread_local_sum += mat4.z * vec4.z;
            thread_local_sum += mat4.w * vec4.w;
        }
       
    }
    float block_sum = blockReduce<SumOp, float>(thread_local_sum);
    if (tid == 0)
    {
        dst[blockIdx.x] = block_sum;
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
//                              
//THREAD_NUMS表示每个block的线程数
template<size_t VECS_PER_THREAD, size_t VEC_SIZE,size_t THREAD_NUMS>
struct DispatchLauncher
{
    template<typename T>
    static void launch( T* mat, T* vec,T* dst,int m,int n)
    {
        dim3 grid(m);
        dim3 block(THREAD_NUMS);
        float time=0.0f;
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop); 
        cudaEventRecord(start,0);
        gemvKernel<VECS_PER_THREAD,VEC_SIZE><<<grid,block>>>(mat,vec,dst,m,n);
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

    for(int i=0;i < N;++i)
    {
        vec[i] = 1.0f;
    }
    for(int i= 0;i<M*N;++i)
    {
        mat[i] =  1.0f;
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

namespace gemv_2
{

//Cols: mat's Cols;
//get the threads needed when prcocessing mat's one row. row/4 when T is fp32, row/8 when T is fp16
template<size_t Cols,typename T>
struct ThreadsPerMatRow
{
    static constexpr size_t value = Cols*sizeof(T)/16;
};
__device__ __forceinline__ float fma(float a, float b, float c)
{
    return a*b+c;
}

__device__ __forceinline__ float4 fma(float a, float4 m,float4 b)
{
    float4 res;
    res.x= gemv_2::fma(a,m.x,b.x);
    res.y= gemv_2::fma(a,m.y,b.y);
    res.z= gemv_2::fma(a,m.z,b.z);
    res.w= gemv_2::fma(a,m.w,b.w);
    return res;
}
__device__ __forceinline__ float add(float a, float b)
{
    return a+b;
}
__device__ __forceinline__ float4 add(float4 a, float4 b)
{
    float4 res;
    res.x = gemv_2::add(a.x,b.x);
    res.y = gemv_2::add(a.y,b.y);
    res.z = gemv_2::add(a.z,b.z);
    res.w = gemv_2::add(a.w,b.w);
    return res;
}


//vec[1,N]*mat[N,M]
template<typename T>
void gemvCPUV2(const T *mat, const T *vec, T *res, int n, int m)
{
   for(int i= 0; i<m;++i)
   {
        res[i] = 0.0f;
        for(int j = 0;j<n;++j)
        {
            res[i] += vec[j]*mat[j+m*i];
        }
   }
}



// vec[1,N]*mat[N,M]
template<size_t THREADS_PER_ROW, size_t THREADS_PER_BLCOK,size_t VEC_SIZE>
__global__ void gemvKernel(float* mat,float*vec,float*dst,int N,int M)
{
    int tid = threadIdx.x;
    //int bid = blockIdx.x;
    // current row in the matrix in the block
    int mat_row = tid / THREADS_PER_ROW;
    // current  position in the row of the matrix in the block
    int mat_index = (tid % THREADS_PER_ROW) * VEC_SIZE;
    // one block sise can deal with rows_per_block rows
    constexpr size_t rows_per_block = THREADS_PER_BLCOK / THREADS_PER_ROW;

    float4 out;
    // intra fma reduction
    for(int row = mat_row; row < N ; row += rows_per_block)
    {
        float4 mat4 = *reinterpret_cast<float4*>(&mat[row*M+mat_index]);
        float  v = vec[row];
        out = gemv_2::fma(v,mat4,out);
    }

    // we nonly need half of the block size*M to store the data
    //constexpr size_t SM_SIZE= M*rows_per_block/2; 
    __shared__ float out_smem[512];
    // inter binary reduction
    for(int row = rows_per_block; row>=2; row/=2)
    {
        int mid = row/2;
        if(mat_row >= mid && mat_row < row)
        {   
            // store the result in the first half of the shared memory
            *reinterpret_cast<float4*>(&out_smem[(mat_row-mid)*M + mat_index])= out;
        }
        __syncthreads();
        if(mat_row < mid)
        {
            // add the second half of the shared memory to the first half
            out = gemv_2::add(*reinterpret_cast<float4*>(&out_smem[mat_row*M+mat_index]),out);
        }
        __syncthreads();
    }

    // the final result is stored in the first row after binary reduction
    if(mat_row == 0)
    {
        *reinterpret_cast<float4*>(&dst[mat_index])= out;
    }                                                                                                                                               

}




template<size_t THREADS_PER_ROW, size_t VEC_SIZE,size_t THREADS_PER_BlOCK>
struct DispatchLauncher
{
    template<typename T>
    static void launch( T* mat, T* vec,T* dst,int n,int m)
    {
        dim3 grid(1);
        dim3 block(THREADS_PER_BlOCK);
        float time=0.0f;
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop); 
        cudaEventRecord(start,0);
        gemvKernel<THREADS_PER_ROW,THREADS_PER_BlOCK,VEC_SIZE><<<grid,block>>>(mat,vec,dst,n,m);
        CUDACHECK(cudaGetLastError());
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time,start,stop);
        std::cout<<"Time: "<<time<<std::endl;
    }
};

//v[1,N]*M[N,M]
template<typename T>
void gemv_kernel(T* mat,T*d_mat,T*vec,T*d_vec,T*dst,T*d_dst)
{

    constexpr size_t N = 2048;
    constexpr size_t M = 256;
    vec = (T*)malloc(N*sizeof(T));
    mat = (T*)malloc(N*M*sizeof(T));
    dst = (T*)malloc(M*sizeof(T));

    cudaMalloc(&d_mat,N*M*sizeof(T));
    cudaMalloc(&d_vec,N*sizeof(T));
    cudaMalloc(&d_dst,M*sizeof(T));

    for(int i=0;i < N;++i)
    {
        vec[i] = 1.5f;
    }
    for(int i= 0;i<M*N;++i)
    {
        mat[i] =  1.0f;
    }

    cudaMemcpy(d_mat,mat,M*N*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec,vec,N*sizeof(T),cudaMemcpyHostToDevice);

    
    constexpr size_t THREAD_Per_Block = 256;
    constexpr size_t VEC_SIZE = Vec<T>::size;
    constexpr size_t THREADS_PER_ROW = gemv_2::ThreadsPerMatRow<M,T>::value;

    DispatchLauncher<THREADS_PER_ROW,VEC_SIZE,THREAD_Per_Block>::template launch(d_mat,d_vec,d_dst,N,M);

    cudaMemcpy(dst,d_dst,M*sizeof(T),cudaMemcpyDeviceToHost);

    T* dst_cpu = (T*)malloc(M*sizeof(T));
    gemvCPUV2(mat,vec,dst_cpu,N,M);
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


}






int main()
{
    float *mat = nullptr;
    float *vec = nullptr;
    float *d_mat = nullptr;
    float *d_vec = nullptr;
    float *d_dst = nullptr;
    float *dst = nullptr;
    //gemv_kernel(mat,d_mat,vec,d_vec,dst,d_dst);
    gemv_2::gemv_kernel(mat,d_mat,vec,d_vec,dst,d_dst);
    return 0;
}