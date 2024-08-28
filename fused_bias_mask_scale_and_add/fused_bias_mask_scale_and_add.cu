#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include <__clang_cuda_builtin_vars.h>
#include <climits>
#include <cstddef>
#include <cstdint>
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
std::vector<U> generatevector(size_t N) {
  std::random_device rd;
  std::mt19937 gen(rd());
  if constexpr (std::is_unsigned<U>::value) {
    std::uniform_int_distribution<U> dis(0, std::numeric_limits<U>::max());
    std::vector<U> vec(N, 0);
    for (size_t i = 0; i < N; ++i) {
      vec[i] = dis(gen);
    }
    return vec;
  } else {
    std::uniform_int_distribution<int> dis(-10000, 10000);
    std::vector<U> vec(N, 0);
    for (size_t i = 0; i < N; ++i) {
      vec[i] = static_cast<U>(dis(gen));
    }
    return vec;
  }
}




//baiseadd + mask+ scale + elementwise add computor
//(x+ baise)*mask*scale + y
template <typename T>
struct FusedBiasMaskScaleAndAddFunctor {

    FusedBiasMaskScaleAndAddFunctor(const uint8_t* mask, float scale, const T* add_val)
        : mask(mask), scale(scale), add_val(add_val) {}
    __device__ __forceinline__ T operator()(T x, T bias, int idx) const {
        return (x + bias) * mask[idx] * scale + add_val[idx];
    } 

    const uint8_t* mask;
    float scale;
    const T* add_val;
};


//basic kernel: (x+ baise)*mask*scale + y
template <typename T,typename OP>
__global__ void FusedBiasMaskScaleAndAdd(T*y,OP op,const T*x,const T * bias,const int N, const int baise_size)
{
    size_t  idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(size_t i = idx;i<N; i+=blockDim.x*gridDim.x)
    {
        T bias_val = bias[i % baise_size];
        y[i] = op(x[i],bias_val,i);
    }
}

//using shared memory to reduce global memory access
template<typename T,typename OP,size_t biasSize>
__global__ void FusedBiasMaskScaleAndAdd_v2(T*y,OP op,const T*x,const T * bias,const int N, const int baise_size)
{
    size_t  idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T bias_shared[biasSize];
    if(threadIdx.x < baise_size)
    {
        bias_shared[threadIdx.x ] = bias[threadIdx.x ];
    }
    __syncthreads();
     
    for(size_t i = idx;i<N/4; i+=blockDim.x*gridDim.x)
    {
        //load 4 float from x
        float4 x_val = reinterpret_cast<const float4*>(x)[i];
        float4 b;
        b.x = op(x_val.x,bias_shared[(i*4)% baise_size],i*4);
        b.y = op(x_val.y,bias_shared[(i*4+1)% baise_size],i*4+1);
        b.z = op(x_val.z,bias_shared[(i*4+2)% baise_size],i*4+2);
        b.w = op(x_val.w,bias_shared[(i*4+3)% baise_size],i*4+3);
        reinterpret_cast<float4*>(y)[i] = b;
    }
}

float test_FusedBiasMaskScaleAndAdd_v1()
{
    const size_t N = 1 << 20;
    const size_t bias_size = 128;
    const size_t size = N * sizeof(float);
    const size_t bias_size_byte = bias_size * sizeof(float);
    const size_t mask_size = N * sizeof(uint8_t);
    const float scale = 0.5f;
    float elapsed_time = 0.0f;


    auto h_x = generatevector<float>(N);
    auto h_y = generatevector<float>(N);
    auto h_bias = generatevector<float>(bias_size);
    auto h_mask = generatevector<uint8_t>(N);
    auto h_add_val = generatevector<float>(N);

    float *d_x, *d_y, *d_bias, *d_add_val;
    uint8_t *d_mask;
    CUDACHECK(cudaMalloc(&d_x, size));
    CUDACHECK(cudaMalloc(&d_y, size));
    CUDACHECK(cudaMalloc(&d_bias, bias_size_byte));
    CUDACHECK(cudaMalloc(&d_mask, mask_size));
    CUDACHECK(cudaMalloc(&d_add_val, size));

    CUDACHECK(cudaMemcpy(d_x, h_x.data(), size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size_byte, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_mask, h_mask.data(), mask_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_add_val, h_add_val.data(), size, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Create the functor and pass device pointers to the constructor!
    FusedBiasMaskScaleAndAddFunctor<float> op(d_mask, scale, d_add_val);
    CudaTimer timer("FusedBiasMaskScaleAndAdd");
    timer.startTiming();
    FusedBiasMaskScaleAndAdd<float, FusedBiasMaskScaleAndAddFunctor<float>><<<grid, block>>>(d_y, op, d_x, d_bias, N, bias_size);
    elapsed_time = timer.stopTiming();
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaMemcpy(h_y.data(), d_y, size, cudaMemcpyDeviceToHost));

    auto y_cpu =generatevector<float>(N);

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = (h_x[i] + h_bias[i % bias_size]) * h_mask[i] * scale + h_add_val[i];
    }
    // Validate the result
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(h_y[i] - y_cpu[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": " << h_y[i] << " != " << y_cpu[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    return elapsed_time;
}

float test_FusedBiasMaskScaleAndAdd_v2()
{
    const size_t N = 1 << 20;
    const size_t bias_size = 128;
    const size_t size = N * sizeof(float);
    const size_t bias_size_byte = bias_size * sizeof(float);
    const size_t mask_size = N * sizeof(uint8_t);
    const float scale = 0.5f;
    float elapsed_time = 0.0f;


    auto h_x = generatevector<float>(N);
    auto h_y = generatevector<float>(N);
    auto h_bias = generatevector<float>(bias_size);
    auto h_mask = generatevector<uint8_t>(N);
    auto h_add_val = generatevector<float>(N);

    float *d_x, *d_y, *d_bias, *d_add_val;
    uint8_t *d_mask;
    CUDACHECK(cudaMalloc(&d_x, size));
    CUDACHECK(cudaMalloc(&d_y, size));
    CUDACHECK(cudaMalloc(&d_bias, bias_size_byte));
    CUDACHECK(cudaMalloc(&d_mask, mask_size));
    CUDACHECK(cudaMalloc(&d_add_val, size));

    CUDACHECK(cudaMemcpy(d_x, h_x.data(), size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size_byte, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_mask, h_mask.data(), mask_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_add_val, h_add_val.data(), size, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    FusedBiasMaskScaleAndAddFunctor<float> op(d_mask, scale, d_add_val);
    CudaTimer timer("FusedBiasMaskScaleAndAdd_v2");
    timer.startTiming();
    FusedBiasMaskScaleAndAdd_v2<float, FusedBiasMaskScaleAndAddFunctor<float>,128><<<grid, block>>>(d_y, op, d_x, d_bias, N, bias_size);
    elapsed_time = timer.stopTiming();
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaMemcpy(h_y.data(), d_y, size, cudaMemcpyDeviceToHost));

    auto y_cpu =generatevector<float>(N);

    for (size_t i = 0; i < N; ++i) {
        y_cpu[i] = (h_x[i] + h_bias[i % bias_size]) * h_mask[i] * scale + h_add_val[i];
    }
    // Validate the result
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(h_y[i] - y_cpu[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": " << h_y[i] << " != " << y_cpu[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    return elapsed_time;
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
void loop_test()
{
    const int n = 10;
    LOOP_TEST(test_FusedBiasMaskScaleAndAdd_v1,n,1);
    LOOP_TEST(test_FusedBiasMaskScaleAndAdd_v2,n,1);

}


int main()
{
    printGPUInfo();
    loop_test();
    return 0;
}

