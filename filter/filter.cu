
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



float test_filter_v1() {

  const int N = 1000000;  
  const int num_bins = 256;
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

    return elapsed_time;
  }
}


void loop_test()
{
    const int n = 10;
    LOOP_TEST(test_filter_v1,n,1);
}


int main()
{
    printGPUInfo();
    loop_test();
    return 0;
}



