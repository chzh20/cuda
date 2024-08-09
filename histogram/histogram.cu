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

// template<typename T, 
//          typename = std::enable_if_t<std::is_integral_v<T>>>
// bool operator == (const Matrix<T> & one,const Matrix<T> &other)  noexcept
// {
//     return one.isEqual(other);
// }

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

// template<typename T, typename C = Matrix<T> >
// class Vector
// {
// public:
//     Vector(size_t N):{ }
//     using vaule_type = T;


// private:
//     C m_matrix;
// }



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

#define BASELINE_TEST(test_func,n,row,col) \
({\
    float elapsed_time = 0.0f;\
    for(int i=0;i<n;++i)\
    {\
        elapsed_time += test_func(row,col);\
    }\
    std::cout << #test_func << " average time: " << (elapsed_time / n) << " ms" << std::endl;\
    elapsed_time/n;\
}) 


template <typename T>
void histogram_cpu(const T* data, size_t N, int* hist, int num_bins) {
    for (size_t i = 0; i < N; ++i) {
        int bin = data[i] % num_bins;
        hist[bin]++;
    }
}



template <typename T>
__global__ void histogram(const T* data, size_t N, int* hist, int num_bins) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int bin = data[tid] % num_bins;
    atomicAdd(&hist[bin], 1);
  }
}

float test_histogram() {

  const int N = 1000000;  
  const int num_bins = 256;
  Matrix<int> data = generateVetcor(N);
  int* d_data;
  int* d_hist;
  int* hist = new int[num_bins];
  memset(hist, 0, num_bins * sizeof(int));
  CUDACHECK(cudaMalloc(&d_data, N * sizeof(int)));
  CUDACHECK(cudaMalloc(&d_hist, num_bins * sizeof(int)));
  CUDACHECK(cudaMemcpy(d_data, data.data_ptr(), N * sizeof(int), cudaMemcpyHostToDevice));
  cudaMemset(d_hist, 0, num_bins * sizeof(int));
  CudaTimer timer;
  timer.startTiming();
  histogram<<<(N + 255) / 256, 256>>>(d_data, N, d_hist, num_bins);
  CUDACHECK(cudaDeviceSynchronize());
  float elapsed_time = timer.stopTiming();
  CUDACHECK(cudaMemcpy(hist, d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaFree(d_data));
  CUDACHECK(cudaFree(d_hist));

  int* hist_cpu = new int[num_bins];
  memset(hist_cpu, 0, num_bins * sizeof(int));
  histogram_cpu(data.data_ptr(),N, hist_cpu,  num_bins);

  for (int i = 0; i < num_bins; ++i) {
    if (hist[i] != hist_cpu[i]) {
      std::cerr << "Mismatch at bin " << i << " expected " << hist_cpu[i]
                << " got " << hist[i] << std::endl;
      break;
    }
  }

  delete[] hist;
  return elapsed_time;
}
void loop_test()
{
    const int n = 10;
    LOOP_TEST(test_histogram,n,1);
}


int main()
{
    printGPUInfo();
    loop_test();
    return 0;
}
