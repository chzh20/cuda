
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
#include "cooperative_groups.h"
#include "cooperative_groups/details/helpers.h"

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

class CudaTimer 

{
private:
    cudaEvent_t start, stop;
    std::string m_kernalName;

public:
    // Constructor
    CudaTimer(const std::string& kernel_name = "") : m_kernalName(kernel_name){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    // Destructor
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Start timing
    void startTiming() {
        cudaEventRecord(start, 0);
    }

    // Stop timing and return elapsed time in milliseconds
    float stopTiming() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        //std::cout<< m_kernalName << " elapsed time: " << milliseconds << " ms" << std::endl;
        return milliseconds;
        

    }
};


void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max Block Dimensions: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Max Grid Dimensions: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << std::endl;
    }
}


template <typename U = int>
static std::vector<U> generatevector(size_t N) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(-10000, 10000);
  std::vector<U> vec(N,0);
  for(int i = 0; i < N; i++){
    vec[i] = static_cast<U>(dis(gen));
  }
  return vec;
}

#define LOOP_TEST(test_func,n,baseline_time) \
{\
    float elapsed_time = 0.0f;\
    for(int i=0;i<n;++i)\
    {\
        elapsed_time += test_func();\
    }\
    std::cout<<#test_func<<" Average elapsed time: "<<elapsed_time/n<<" ms"<<std::endl;\
    if(baseline_time > 0)\
    {\
        std::cout<<"Speedup: "<<baseline_time/(elapsed_time/n)<<std::endl;\
    }\
}

template <typename T>
int filter(T* dest, T *src,int N)
{
    int count = 0;
    for(int i = 0; i < N; i++){
        if(src[i] > 0){
            dest[count++] = src[i];
        }
    }
    return count;
}


template <typename T>
__global__ void filter_v1(T* dest, T *src,T*counter,int N)
{
    static_assert(std::is_integral<T>::value, "T must be an integral type");
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N && src[tid] > 0){
        dest[atomicAdd(counter,1)] = src[tid];
    }
}

template <typename T,size_t blocksize>
__global__ void filter_v2(T* dest, T *src,T*counter,int N)
{
    __shared__ int s_counter;
    __shared__ int offset;
    int pos = 0;
    s_counter = 0;
    offset = 0;
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = tid<N?src[tid]:0;
    if(val > 0){
        pos = atomicAdd(&s_counter,1);
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        offset = atomicAdd(counter,s_counter);
    }
    __syncthreads();
    if(val > 0)
    {   
        dest[offset+pos] = val;
    }
}

template <typename T,size_t blocksize>
__global__ void filter_v3(T* dest, T *src,T*counter,int N)
{
    __shared__ int s_counter;
    __shared__ int offset;
    __shared__ int s_dest[blocksize];
    int pos = 0;
    s_counter = 0;
    offset = 0;
    for(int i = threadIdx.x; i < blocksize; i+=blockDim.x)
    {
        s_dest[i] = 0;
    }
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = src[tid];
    if(tid <N && val > 0){
        pos = atomicAdd(&s_counter,1);
        s_dest[pos] = val;
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        offset = atomicAdd(counter,s_counter);
    }
    __syncthreads();
    for(int i = threadIdx.x; i < s_counter; i+=blockDim.x)
    {
        dest[offset+i] = s_dest[i];
    }
}

__device__ int atomicAggInc(int *ctr) {
 
  //__activemask 返回当前warp中活跃线程的掩码
  unsigned int active = __activemask();
  //leader表示当前warp中第一个src[threadIdx.x]>0的threadIdx.x
  int leader = __ffs(active) - 1;
  //change表示当前warp中src[threadIdx.x]>0的数量
  int change = __popc(active);//warp mask中为1的数量
  
  //计算lane_mask_lt是一个小于当前线程id的mask，比如线程id为5，那么lane_mask_lt为00011111
  //包括非活跃的线程。使用AND操作符将__lanemask_lt与活跃线程的掩码相结合，以便在warp中计算线程的排名。
  int lane_mask_lt;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));

  // rank表示当前活跃线程在warp中的排名
  unsigned int rank = __popc(active & lane_mask_lt); // 比当前线程id小且值为1的mask之和
  int warp_res;
  // wrap_res表示当前warp的全局offset.
  if(rank == 0)//leader thread of every warp
    warp_res = atomicAdd(ctr, change);//compute global offset of warp
  //将leader线程的warp_res广播给warp中的其他线程
  warp_res = __shfl_sync(active, warp_res, leader);//broadcast warp_res of leader thread to every active thread
  return warp_res + rank; // global offset + local offset = final offset，即L91表示的atomicAggInc(nres), 为src[i]的最终的写入到dst的位置
}

template <typename T,size_t blocksize>
__global__ void filter_warp_k(T *dst, T *nres, T *src, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i >= n)
    return;
  if(src[i] > 0) // 过滤出src[i] > 0 的线程，比如warp0里面只有0号和1号线程的src[i]>0，那么只有0号和1号线程运行L91，对应的L72的__activemask()为110000...00
    // 以上L71函数计算当前thread负责数据的全局offset
    dst[atomicAggInc(nres)] = src[i];
}




float test_filter_v1() {

  const int N = 1000000;
  std::vector<int> data = generatevector(N);
  int  counter = 0;
  int* d_data;
  int* d_dest;
  int* d_counter;
  CUDACHECK(cudaMalloc(&d_data, N * sizeof(int)));
  CUDACHECK(cudaMalloc(&d_dest, N * sizeof(int)));
  CUDACHECK(cudaMalloc(&d_counter,   sizeof(int)));
  CUDACHECK(cudaMemcpy(d_data, data.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_counter, &counter,  sizeof(int), cudaMemcpyHostToDevice));
  cudaMemset(d_dest, 0, N * sizeof(int));
  CudaTimer timer;
  timer.startTiming();
  filter_v1<<<(N + 255) / 256, 256>>>(d_dest, d_data, d_counter, N);
  CUDACHECK(cudaDeviceSynchronize());
  float elapsed_time = timer.stopTiming();
  CUDACHECK(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> result(counter,0);

  CUDACHECK(cudaMemcpy(result.data(), d_dest, counter * sizeof(int), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaFree(d_data));
  CUDACHECK(cudaFree(d_dest));
  CUDACHECK(cudaFree(d_counter));

  std::vector<int> result_cpu(N,0);
  int counter_cpu = filter(result_cpu.data(),data.data(),N);
  if(counter != counter_cpu){
        std::cerr<<"Mismatch at counter "<<counter<<" expected "<<counter_cpu<<std::endl;
        return  elapsed_time;
  }

  std::sort(result.begin(), result.end());
  std::sort(result_cpu.begin(), result_cpu.begin() + counter_cpu);
  for (int i = 0; i < counter_cpu; i++) {
    if (result[i] != result_cpu[i]) {
      std::cerr << "Mismatch at " << i << " expected " << result_cpu[i]
                << " got " << result[i] << std::endl;
    }


  }
  return elapsed_time;
}


float test_filter_v2() {

  const int N = 1000000;  
  std::vector<int> data = generatevector(N);
  int  counter = 0;
  int* d_data;
  int* d_dest;
  int* d_counter;
  CUDACHECK(cudaMalloc(&d_data, N * sizeof(int)));
  CUDACHECK(cudaMalloc(&d_dest, N * sizeof(int)));
  CUDACHECK(cudaMalloc(&d_counter,   sizeof(int)));
  CUDACHECK(cudaMemcpy(d_data, data.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_counter, &counter,  sizeof(int), cudaMemcpyHostToDevice));
  cudaMemset(d_dest, 0, N * sizeof(int));
  CudaTimer timer;
  timer.startTiming();
  filter_v2<int,256><<<(N + 255) / 256, 256>>>(d_dest, d_data, d_counter, N);
  CUDACHECK(cudaDeviceSynchronize());
  float elapsed_time = timer.stopTiming();
  CUDACHECK(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> result(counter,0);

  CUDACHECK(cudaMemcpy(result.data(), d_dest, counter * sizeof(int), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaFree(d_data));
  CUDACHECK(cudaFree(d_dest));
  CUDACHECK(cudaFree(d_counter));

  std::vector<int> result_cpu(N,0);
  int counter_cpu = filter(result_cpu.data(),data.data(),N);
  if(counter != counter_cpu){
        std::cerr<<"Mismatch at counter "<<counter<<" expected "<<counter_cpu<<std::endl;
        return  elapsed_time;
  }

  std::sort(result.begin(), result.end());
  std::sort(result_cpu.begin(), result_cpu.begin() + counter_cpu);
  for (int i = 0; i < counter_cpu; i++) {
    if (result[i] != result_cpu[i]) {
      std::cerr << "Mismatch at " << i << " expected " << result_cpu[i]
                << " got " << result[i] << std::endl;
    }

    
  }

  return elapsed_time;
}


float test_filter_v3() {

  const int N = 1000000;  
  std::vector<int> data = generatevector(N);
  int  counter = 0;
  int* d_data;
  int* d_dest;
  int* d_counter;
  CUDACHECK(cudaMalloc(&d_data, N * sizeof(int)));
  CUDACHECK(cudaMalloc(&d_dest, N * sizeof(int)));
  CUDACHECK(cudaMalloc(&d_counter,   sizeof(int)));
  CUDACHECK(cudaMemcpy(d_data, data.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_counter, &counter,  sizeof(int), cudaMemcpyHostToDevice));
  cudaMemset(d_dest, 0, N * sizeof(int));
  CudaTimer timer;
  timer.startTiming();
  filter_v3<int,256><<<(N + 255) / 256, 256>>>(d_dest, d_data, d_counter, N);
  CUDACHECK(cudaDeviceSynchronize());
  float elapsed_time = timer.stopTiming();
  CUDACHECK(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> result(counter,0);

  CUDACHECK(cudaMemcpy(result.data(), d_dest, counter * sizeof(int), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaFree(d_data));
  CUDACHECK(cudaFree(d_dest));
  CUDACHECK(cudaFree(d_counter));

  std::vector<int> result_cpu(N,0);
  int counter_cpu = filter(result_cpu.data(),data.data(),N);
  if(counter != counter_cpu){
        std::cerr<<"Mismatch at counter "<<counter<<" expected "<<counter_cpu<<std::endl;
        return  elapsed_time;
  }

  std::sort(result.begin(), result.end());
  std::sort(result_cpu.begin(), result_cpu.begin() + counter_cpu);
  for (int i = 0; i < counter_cpu; i++) {
    if (result[i] != result_cpu[i]) {
      std::cerr << "Mismatch at " << i << " expected " << result_cpu[i]
                << " got " << result[i] << std::endl;
    }

    
  }

  return elapsed_time;
}

float test_filter_v4(){
  const int N = 1000000;  
  std::vector<int> data = generatevector(N);
  int  counter = 0;
  int* d_data;
  int* d_dest;
  int* d_counter;
  CUDACHECK(cudaMalloc(&d_data, N * sizeof(int)));
  CUDACHECK(cudaMalloc(&d_dest, N * sizeof(int)));
  CUDACHECK(cudaMalloc(&d_counter,   sizeof(int)));
  CUDACHECK(cudaMemcpy(d_data, data.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_counter, &counter,  sizeof(int), cudaMemcpyHostToDevice));
  cudaMemset(d_dest, 0, N * sizeof(int));
  CudaTimer timer;
  timer.startTiming();
  filter_warp_k<int,256><<<(N + 255) / 256, 256>>>(d_dest, d_counter,d_data, N);
  CUDACHECK(cudaDeviceSynchronize());
  float elapsed_time = timer.stopTiming();
  CUDACHECK(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> result(counter,0);

  CUDACHECK(cudaMemcpy(result.data(), d_dest, counter * sizeof(int), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaFree(d_data));
  CUDACHECK(cudaFree(d_dest));
  CUDACHECK(cudaFree(d_counter));

  std::vector<int> result_cpu(N,0);
  int counter_cpu = filter(result_cpu.data(),data.data(),N);
  if(counter != counter_cpu){
        std::cerr<<"Mismatch at counter "<<counter<<" expected "<<counter_cpu<<std::endl;
        return  elapsed_time;
  }

  std::sort(result.begin(), result.end());
  std::sort(result_cpu.begin(), result_cpu.begin() + counter_cpu);
  for (int i = 0; i < counter_cpu; i++) {
    if (result[i] != result_cpu[i]) {
      std::cerr << "Mismatch at " << i << " expected " << result_cpu[i]
                << " got " << result[i] << std::endl;
    }

    
  }

  return elapsed_time;
}


void loop_test()
{
    const int n = 10;
    LOOP_TEST(test_filter_v1,n,1);
    LOOP_TEST(test_filter_v2,n,1);
    LOOP_TEST(test_filter_v3,n,1);
}


int main()
{
    //printGPUInfo();
    loop_test();
    return 0;
}



