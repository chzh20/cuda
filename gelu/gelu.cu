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

template<typename T>
void gelu_cpu(T* odata,T* idata,size_t N)
{
    for(size_t i = 0;i<N;++i)
    {
        //gelu(x)= 0.5*x*(1+tanh(sqrt(2/pi)(x+0.044715*x^3))
        const T x = idata[i];
        const T cdf = T(0.5) * (T(1.0) + tanh((T(0.797884) * (x + T(0.044715) * x * x * x))));
        odata[i] = x * cdf;
    }
}




template<typename T>
__global__ void gelu_kernel(T* odata, const T* idata, size_t N)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        const T x = idata[idx];
        const T cdf = T(0.5) * (T(1.0) + tanh((T(0.797884) * (x + T(0.044715) * x * x * x))));
        odata[idx] = x * cdf;
    }
}





template<typename T>
void gelu_gpu(T* odata, T* idata, size_t N)
{
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    gelu_kernel<<<blocks, threadsPerBlock>>>(odata, idata, N);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
}


template<typename T,size_t Size>
struct alignas(sizeof(T)*Size) AlignedVector
{
    T data[Size];
    __host__ __device__ inline const T& operator[](size_t i) const { return data[i]; }
    __host__ __device__ inline T& operator[](size_t i) { return data[i]; }

};

template<typename T>
struct GeluFunctor
{
    static constexpr T sqrt_2_over_pi = static_cast<T>(0.79788456080286541);
    static constexpr T a =  static_cast<T>(0.044715);

    __device__ GeluFunctor() = default;

    __device__ __forceinline__ T operator()(T x) const
    {
       const T cdf = T(0.5) * (T(1.0) + tanh(sqrt_2_over_pi * (x + a * x * x * x)));
        return x * cdf;
    }
};

__device__ __forceinline__ float TanhApprox(float x)
{
#if __CUDA_ARCH__ >= 800
    float r;
    asm("tanh.approx.f32 %0,%1;" : "=f"(r) : "f"(x));
    return r;
#else
    return tanhf(x);
#endif

}


template<>
struct GeluFunctor<half>
{
    static constexpr float sqrt_2_over_pi= 0.79788456080286541f;
    static constexpr float a = 0.044715;
   
    __device__ GeluFunctor() = default;


    __device__ __forceinline__ half operator()(half x) const
    {
#if __CUDA_ARCH__ >= 800
        const float tanh_in = __half2float(__float2half_rn(sqrt_2_over_pi) * (x + __float2half_rn(a) * x * x * x));
        const float tanh_out = TanhApprox(tanh_in);
        const half cdf = 0.5h * (1.0h + __float2half_rn(tanh_out));
        return x * cdf;

#else
     GeluFunctor <float> float_gelu;
     return static_cast<half>(float_gelu(static_cast<float>(x)));
#endif

    }
};


template<size_t VecSize>
__global__ void FP16GeluKernel(half* odata,const half* idata,size_t N)
{
    //向量化 load & store
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
    int stride = blockDim.x * gridDim.x * VecSize;
    GeluFunctor<half> gelu;
    __half y_reg[VecSize];

    using ArrT = AlignedVector<__half, VecSize>;
    for(; offset < N; offset += stride)
    {
        const __half* in = idata + offset;
        __half* out = odata + offset;

        if(VecSize == 1)
        {
            y_reg[0] = gelu(in[0]);
        }
        else
        {
            #pragma unroll
            for(int i = 0;i < VecSize;++i)
            {
                y_reg[i] = gelu(in[i]);
            }
        }

        *reinterpret_cast<ArrT*>(out) = *reinterpret_cast<ArrT*>(y_reg);
    }
}


template<typename T>
float test_gelu()
{
    const size_t N = 1 << 20;  // 大约100万个元素
    std::vector<T> h_input = generatevector<T>(N);
    std::vector<T> h_output_cpu(N);
    std::vector<T> h_output_gpu(N);

    // 分配GPU内存
    T *d_input, *d_output;
    CUDACHECK(cudaMalloc(&d_input, N * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_output, N * sizeof(T)));

    // 将输入数据复制到GPU
    CUDACHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    // CPU计算
    gelu_cpu(h_output_cpu.data(), h_input.data(), N);

    // GPU计算
    CudaTimer timer("GELU GPU");
    timer.startTiming();
    gelu_gpu(d_output, d_input, N);
    float elapsed_time = timer.stopTiming();

    // 将结果复制回主机
    CUDACHECK(cudaMemcpy(h_output_gpu.data(), d_output, N * sizeof(T), cudaMemcpyDeviceToHost));

    // 验证结果
    double max_error = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double error = std::abs(static_cast<double>(h_output_cpu[i]) - static_cast<double>(h_output_gpu[i]));
        max_error = std::max(max_error, error);
    }
    std::cout << "最大误差: " << max_error << std::endl;

    // 释放GPU内存
    CUDACHECK(cudaFree(d_input));
    CUDACHECK(cudaFree(d_output));

    return elapsed_time;
}



void looptest()
{
    std::cout << "测试单精度GELU:" << std::endl;
    LOOP_TEST(test_gelu<float>, 10, 1);

    std::cout << "\n测试双精度GELU:" << std::endl;
    LOOP_TEST(test_gelu<double>, 10, 1);
}



int main()
{
    printGPUInfo();
    looptest();
    return 0;
}

