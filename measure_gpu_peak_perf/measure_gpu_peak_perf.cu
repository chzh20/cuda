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
#include<cassert>
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



template<typename T = int>
class Matrix
{
private:
    size_t m_row;
    size_t m_col;
    std::vector<T> m_data;
public:
    using value_type = T;
    enum class MatrixType
    {
        RowMajor,
        ColMajor
    };
    Matrix(size_t row,size_t col)  noexcept:m_row(row),m_col(col),m_data(row*col,T{}){}
    Matrix(size_t row,size_t col,std::vector<T> data) noexcept:m_row(row),m_col(col),m_data(data){}
    Matrix(size_t row,size_t col,T* data) noexcept:m_row(row),m_col(col),m_data(data,data+row*col){}
    Matrix(size_t row,size_t col,const T& value) noexcept:m_row(row),m_col(col),m_data(row*col,value){}
    Matrix(const Matrix& other)noexcept:m_row(other.m_row),m_col(other.m_col),m_data(other.m_data){}
    Matrix(Matrix&& other)noexcept:m_row(other.m_row),m_col(other.m_col),m_data(std::move(other.m_data)){}
    Matrix<T> & operator =(const Matrix<T> & matrix) noexcept
    {
        if(this != &matrix)
        {
            m_row = matrix.m_row;
            m_col = matrix.m_col;
            m_data = matrix.m_data;
        }
        return *this;
    }
    Matrix<T> & operator =(Matrix<T> && matrix) noexcept
    {
        if(this != &matrix)
        {
            m_row = matrix.m_row;
            m_col = matrix.m_col;
            m_data = std::move(matrix.m_data);
        }
        return *this;
    }
   
    const T& operator() (size_t row,size_t col) const noexcept
    {
        return m_data[row*m_col+col];
    }
    T& operator() (size_t row,size_t col) noexcept
    {
        // if(row >=m_row || col>=m_col )
        // {
        //     throw std::out_of_range("Matrix subscript out of range");
        // }
        return m_data[row*m_col+col];
    }
    const T& operator[] (size_t index) const noexcept
    {
        // if(index >=m_row*m_col )
        // {
        //     throw std::out_of_range("Matrix subscript out of range");
        // }
        return m_data[index];
    }
    T& operator[] (size_t index) noexcept
    {
        // if(index >=m_row*m_col)
        // {
        //     throw std::out_of_range("Matrix subscript out of range");
        // }
        return m_data[index];
    }
    std::vector<T> data() const noexcept
    {
        return m_data;
    }
    T* data_ptr() noexcept
    {
        return m_data.data();
    }
    const T* data_ptr() const noexcept
    {
        return m_data.data();
    }
   
    
    void printfMatrix() const noexcept
    {
        for(size_t i = 0;i<m_row;++i)
        {
            for(size_t j = 0;j<m_col;++j)
            {
                std::cout<<m_data[i*m_col+j]<<" ";
            }
            std::cout<<std::endl;
        }
    }
    
    bool isEqual(const Matrix<T>& other) const noexcept
    {
        if(m_row != other.m_row || m_col != other.m_col)
        {
            return false;
        }
        for(size_t i = 0;i<m_row;++i)
        {
            for(size_t j = 0;j<m_col;++j)
            {
                if(m_data[i*m_col+j] != other.m_data[i*m_col+j])
                {
                    return false;
                }
            }
        }
        return true;
    }
    size_t row() const noexcept
    {
        return m_row;
    }
    size_t col() const noexcept
    {
        return m_col;
    }
    template<typename U>
    friend bool operator == (const Matrix<U> & one,const Matrix<U> & other) noexcept;

    Matrix<T> transpose() const noexcept
    {
        Matrix<T> result(m_col,m_row);
        for(size_t i = 0;i<m_row;++i)
        {
            for(size_t j = 0;j<m_col;++j)
            {
                result(j,i) = m_data[i*m_col+j];
            }
        }
        return result;
    }

    template<typename U>
    void deepCopy(const Matrix<U>& other) noexcept
    {
        m_row = other.row();
        m_col = other.col();
        m_data.resize(m_row*m_col);
        for(size_t i = 0;i<m_row;++i)
        {
            for(size_t j = 0;j<m_col;++j)
            {
                m_data[i*m_col+j] = other(i,j);
            }
        }
    }

};

template<typename T>
bool operator == (const Matrix<T> & one,const Matrix<T> &other)  noexcept
{
     float epsilon = 1e-5;
     if (one.row() != other.row() || one.col() != other.col()) {
       return false;
     }
     for (size_t i = 0; i < one.row(); ++i) {
       for (size_t j = 0; j < one.col(); ++j) {
         if (std::abs(one(i, j) - other(i, j)) > epsilon) {
           return false;
         }
       }
     }
     return true;
}

template<typename U>
bool operator != (const Matrix<U> & one,const Matrix<U> &other)  noexcept
{
    return !(one==other);
}

template <typename U = int>
static Matrix<U> generateMatrix(size_t row, size_t col) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 10000);
  Matrix<U> matrix(row, col);
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      matrix(i, j) = static_cast<U>(dis(gen));
    }
  }
  return matrix;
}
template <typename U = int>
static Matrix<U> generateVetcor(size_t N) {
  return generateMatrix(N,1);
}

template<size_t LOOPSIZE=4>
__global__  void fma_kernel(const float* A, const float* B, float* D,int *start,int *stop,size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start_time =0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start_time)::"memory");
    float res= 0.0f;
    //if(idx < N)
    //{
        for(size_t i = 0;i<LOOPSIZE;++i)
        {
            res = A[idx] * B[idx] + res;
            //res = A[idx] * B[idx] + res;
            //res = A[idx] * B[idx] + res;
            //res = A[idx] * B[idx] + res;
        }
    //}
    //sync 
    asm volatile("bar.sync 0;");
    int stop_time =0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop_time)::"memory");
    start[idx] = start_time;
    stop[idx] = stop_time;
    D[idx] = res;
}



/*
To calculate the peak FLOPS (Floating Point Operations Per Second) of a GPU, you can use the following formula:

Peak FLOPS = Number of Cores * Clock Speed (in GHz)} * FLOP per Cycle

Where:
- **Number of Cores**: The total number of processing cores in the GPU.
- **Clock Speed**: The operating frequency of the GPU in gigahertz (GHz).
- **FLOP per Cycle**: The number of floating-point operations that can be performed per clock cycle. This typically depends on the architecture of the GPU (e.g., single-precision vs. double-precision).
### Example Calculation
1. **Number of Cores**: 2560
2. **Clock Speed**: 1.5 GHz
3. **FLOP per Cycle**: 2 (for single-precision)

*/





float test_fma()
{
    size_t N = 1024;
    constexpr size_t LOOPSIZE = 1000;
    size_t size = N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_D = (float*)malloc(size);
    int *h_start = (int*)malloc(N*sizeof(int));
    int *h_stop = (int*)malloc(N*sizeof(int));
    for(size_t i = 0;i<N;++i)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_D = nullptr;
    int *d_start = nullptr;
    int *d_stop = nullptr;

    CUDACHECK(cudaMalloc(&d_A,size));
    CUDACHECK(cudaMalloc(&d_B,size));
    CUDACHECK(cudaMalloc(&d_D,size));
    CUDACHECK(cudaMalloc(&d_start,N*sizeof(int)));
    CUDACHECK(cudaMalloc(&d_stop,N*sizeof(int)));
    CUDACHECK(cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice));
    CudaTimer timer("fma_kernel");
    timer.startTiming();
    fma_kernel<<<1,1024>>>(d_A,d_B,d_D,d_start,d_stop,N);
    CUDACHECK(cudaDeviceSynchronize());
    float elapsed_time = timer.stopTiming();
    CUDACHECK(cudaMemcpy(h_D,d_D,size,cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_start,d_start,N*sizeof(int),cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_stop,d_stop,N*sizeof(int),cudaMemcpyDeviceToHost));
    float avg_time = 0.0f;
    for(size_t i = 0;i<N;++i)
    {
        avg_time += (h_stop[i] - h_start[i]);
    }
    avg_time = avg_time/N;
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    // Calculate the peak FLOPS
    // Peak FLOPS = Number of Cores * Clock Speed (in GHz) * FLOP per Cycle
    float clockRate = props.clockRate * 1e-6f; // 转换为 GHz
    int numSMs = props.multiProcessorCount;
    float flop =(LOOPSIZE*2*1024)/avg_time;
    std::cout<<"GPU Peak Performance: "<<flop*clockRate*numSMs*1e-3f<<" TFLOPS"<<std::endl;
    free(h_A);
    free(h_B);
    free(h_D);
    free(h_start);
    free(h_stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    cudaFree(d_start);
    cudaFree(d_stop);
    return elapsed_time;
}
int main()
{
    printGPUInfo();
    float elapsed_time = test_fma();
    std::cout<<"Elapsed time: "<<elapsed_time<<" ms"<<std::endl;
    return 0;
}