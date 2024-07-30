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



using data_type = int;

template<typename T>
__global__ void add_kernel(T* input,T* output,size_t N)
{
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if(id ==0)
   {
     printf("add kernel is call\n");
   }
   if(id<N)
   {
        output[id] = input[id]+id;
   }
}

template<typename T>
void add_kernel_host(T* input,T* output,size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        output[i] = input[i] + i;
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
    std::vector<T>& data() const noexcept
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

void testMatrix()
{
    Matrix matrix = generateMatrix(10,10);
    matrix.printfMatrix();
}



template<typename T>
__global__ void copyMatrix(T* odata,T* idata,size_t row,size_t col)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIndex < col && yIndex < row) {
        odata[yIndex * col + xIndex] = idata[yIndex * col + xIndex];
    }

}
template<typename T>
__global__ void transposeMatrix_base(T* odata,T* idata,size_t row,size_t col)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIndex < col && yIndex < row) {
        odata[xIndex * row + yIndex] = idata[yIndex * col + xIndex];
    }

}



const int TILE_DIMX = 32;
const int TILE_DIMY = 32;
/*
    transpose matrix using shared memory

*/
template<typename T = float>
__global__ void transposeMatrix_shared(T* odata,T* idata,size_t row,size_t col)
{
    __shared__ T tile[TILE_DIMY][TILE_DIMX+1];
    
    
    // global  thread index in grids;
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;


    //global thread index in matrix(row,col)
    int index_in = yIndex * col + xIndex;
    
     // thread index in block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    //transpose block. reverse the x and y index
    int irow = tid / blockDim.y;
    int icol = tid % blockDim.x;

    //global thread index in transposed grid blocks.
    int yIndex_t = blockIdx.y * blockDim.y + icol;
    int xIndex_t = blockIdx.x * blockDim.x + irow;

    //global thread index in transposed matrix(col,row)
    int index_out_t = xIndex_t * row + yIndex_t;

    //copy data from global memory to shared memory
    if(xIndex < col && yIndex < row)
    {
        tile[threadIdx.y][threadIdx.x] = idata[index_in];

    }
    __syncthreads();
    if(xIndex_t < row && yIndex_t < col)
    {
        odata[index_out_t] = tile[icol][irow];
    }
}

/*
    transpose matrix using stride.
    one thread deal with mutliple elements.
    reference: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
*/
template<typename T = float>
__global__ void transposeMatrix_stride(T* odata,T* idata,size_t row,size_t col)
{

    size_t global_index = blockDim.x*blockIdx.x + threadIdx.x;
    for(size_t i = global_index;i<row*col;i+=blockDim.x*gridDim.x)
    {
        size_t xIndex = i / row;
        size_t yIndex = i % row;
        odata[xIndex*row+yIndex] = idata[yIndex*col+xIndex];
    }

}


template<typename T,size_t TILEDIM =32>
__global__ void transposeMatrix_shared_2(T * odata, T*idata,size_t row,size_t col)
{
    __shared__ T tile[TILEDIM][TILEDIM+1];
    int  x = blockIdx.x * TILEDIM + threadIdx.x;
    int  y = blockIdx.y * TILEDIM + threadIdx.y;

    if(x<col && y<row)
    {
        tile[threadIdx.y][threadIdx.x] = idata[y*col+x];
    }
    __syncthreads();

    x = blockIdx.y * TILEDIM + threadIdx.x;
    y = blockIdx.x * TILEDIM + threadIdx.y;

    if(x<row && y<col)
    {
        odata[y*row+x] = tile[threadIdx.x][threadIdx.y];
    }
}


// using shared memory and unroll the loop
// use less threads in a block and each thread deal with multiple elements.
// in this case, tile size is 32*32, and block size is 32*8;
// each thread deals with 4 elements.
//https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

template<typename T,size_t TILEDIM = 32, size_t UNROLL_FACTOR = 4>
__global__ void transposeMatrix_shared_unroll(T* odata, T* idata, size_t row, size_t col)
{
    __shared__ T tile[TILEDIM][TILEDIM+1];

    int x = blockIdx.x *TILEDIM + threadIdx.x;
    int y = blockIdx.y *TILEDIM + threadIdx.y;

    #pragma unroll
    for(int i=0; i< UNROLL_FACTOR; ++i)
    { 
        if(x<col && y+i<row)
        {
            tile[threadIdx.y+i][threadIdx.x] = idata[(y+i)*col+x];
        }
       
    }

    __syncthreads();
    x = blockIdx.y * TILEDIM + threadIdx.x;
    y = blockIdx.x * TILEDIM + threadIdx.y;
    #pragma unroll
    for(int i=0; i< UNROLL_FACTOR; ++i)
    { 
        if(x<row && y+i<col)
        {
            odata[(y+i)*row + x] = tile[threadIdx.x][threadIdx.y+i];
        }
        
    }
}


template<typename T> 
float  cublasTransposeMatrix(const Matrix<T>& inputMatrix, Matrix<T>& outputMatrix) {
    // Ensure the output matrix has the correct dimensions
    if (outputMatrix.row() != inputMatrix.col() || outputMatrix.col() != inputMatrix.row()) {
        throw std::invalid_argument("Output matrix dimensions do not match the transposed dimensions of the input matrix.");
    }

    // Get the dimensions of the input matrix
    size_t rows = inputMatrix.row();
    size_t cols = inputMatrix.col();

    // Allocate device memory
    T* d_input = nullptr;
    T* d_output = nullptr;
    CUDACHECK(cudaMalloc(&d_input, rows * cols * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_output, rows * cols * sizeof(T)));

    // Copy data from host to device
    CUDACHECK(cudaMemcpy(d_input, inputMatrix.data_ptr(), rows * cols * sizeof(T), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform the matrix transpose using cuBLAS
    const T alpha = 1.0f;
    const T beta = 0.0f;
    CudaTimer timer("cublasSgeam");
    timer.startTiming();
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, cols, rows, &alpha, d_input, rows, &beta, nullptr, cols, d_output, cols);
    cudaDeviceSynchronize();
    float elapsed_time = timer.stopTiming();

    // Copy the transposed matrix back to the host
    CUDACHECK(cudaMemcpy(outputMatrix.data_ptr(), d_output, rows * cols * sizeof(T), cudaMemcpyDeviceToHost));

    // Clean up
    CUDACHECK(cudaFree(d_input));
    CUDACHECK(cudaFree(d_output));
    cublasDestroy(handle);
    return elapsed_time;
}

float test_cublasTransposeMatrix(size_t row =1024,size_t col =1024)
{
    float elapsed_time = 0.0f;
    using value_type = float;
    auto matrix = generateMatrix<value_type>(row,col);
    Matrix<value_type>  result(matrix.col(),matrix.row());

    elapsed_time = cublasTransposeMatrix(matrix,result);
    Matrix<value_type> cpu_result = matrix.transpose();
    if(result != cpu_result)
    {
        std::cout<<"cublasTransposeMatrix  is incorrect!"<<std::endl;
    }
    return elapsed_time;
}

template<typename F,typename ... Args>
auto  test_kernel(F kernel,size_t row=1024,size_t col =1024,Args... args)
{
    float elapsed_time = 0.0f;
    auto matrix = generateMatrix<float>(row,col);
    Matrix<float>  result(matrix.row(),matrix.col());

    using VauleType = decltype(matrix)::value_type;
    size_t byteSize = matrix.row() * matrix.col() * sizeof(VauleType);

    VauleType* dev_idata;
    VauleType* dev_odata;

    CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&dev_idata),byteSize));
    CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&dev_odata),byteSize));
    CUDACHECK(cudaMemcpy(dev_idata,matrix.data_ptr(),byteSize,cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32,32);
    dim3 blocksPerGrid((matrix.col() + threadsPerBlock.x -1) /threadsPerBlock.x,
                       (matrix.row() + threadsPerBlock.y -1) /threadsPerBlock.y);
    CudaTimer timer("kernel");
    timer.startTiming();
    kernel<<<blocksPerGrid,threadsPerBlock>>>(dev_odata,dev_idata,matrix.row(),matrix.col(),args...);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
    elapsed_time = timer.stopTiming();
    CUDACHECK(cudaMemcpy(result.data_ptr(),dev_odata,byteSize,cudaMemcpyDeviceToHost));
    CUDACHECK(cudaFree(dev_idata));
    CUDACHECK(cudaFree(dev_odata));
    return std::make_tuple(matrix,result,elapsed_time);
}


float test_copyMatrix(size_t row =1024,size_t col =1024)
{
    auto[matrix,result,elapsed_time] = test_kernel(copyMatrix<float>,row,col);
    if(result != matrix)
    {
        std::cout<<"Copy Matrix is incorrect!"<<std::endl;
    }
    return elapsed_time;
}




float  test_transpose_base(size_t row =1024,size_t col =1024)
{
    auto[matrix,result,elapsed_time] = test_kernel(transposeMatrix_base<float>,row,col);
    if(result != matrix.transpose())
    {
        std::cout<<"transposeMatrix_base  is Incorrect!"<<std::endl;
    }
    return elapsed_time;
}



float test_transpose_shared(size_t row =1024,size_t col =1024)
{   
    auto [OringalMatrix,result,elapsed_time]= test_kernel(transposeMatrix_shared<float>,row,col);
    if(result != OringalMatrix.transpose())
    {
        std::cout<<"transposeMatrix_shared  is Incorrect!"<<std::endl;
    }
    return elapsed_time;
}

float test_transpose_shared_2(size_t row =1024,size_t col =1024)
{   
    auto [OringalMatrix,result,elapsed] = test_kernel(transposeMatrix_shared_2<float>,row,col);
    if(result != OringalMatrix.transpose())
    {
        std::cout<<"transposeMatrix_shared_2  is Incorrect!"<<std::endl;
    }
    return elapsed;
}


float  test_transposeMatrix_shared_unroll(size_t row =1024,size_t col =1024)
{
    float elapsed_time = 0.0f;
    auto matrix = generateMatrix<float>(row,col);
    Matrix<float>  result(matrix.row(),matrix.col());

    using VauleType = decltype(matrix)::value_type;
    size_t byteSize = matrix.row() * matrix.col() * sizeof(VauleType);

    VauleType* dev_idata;
    VauleType* dev_odata;

    CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&dev_idata),byteSize));
    CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&dev_odata),byteSize));
    CUDACHECK(cudaMemcpy(dev_idata,matrix.data_ptr(),byteSize,cudaMemcpyHostToDevice));

    const int TILE_DIM = 32;
    dim3 threadsPerBlock(TILE_DIM,8);
    dim3 blocksPerGrid((matrix.col() + TILE_DIM -1) /TILE_DIM,
                       (matrix.row() + TILE_DIM -1) /TILE_DIM);
    CudaTimer timer("transposeMatrix_shared_unroll");
    timer.startTiming();
    transposeMatrix_shared_unroll<<<blocksPerGrid,threadsPerBlock>>>(dev_odata,dev_idata,matrix.row(),matrix.col());
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
    elapsed_time = timer.stopTiming();
    CUDACHECK(cudaMemcpy(result.data_ptr(),dev_odata,byteSize,cudaMemcpyDeviceToHost));
    CUDACHECK(cudaFree(dev_idata));
    CUDACHECK(cudaFree(dev_odata));
    return elapsed_time;
}



float test_transpose_stride(size_t row =1024,size_t col =1024)
{   
    //auto [OringalMatrix,result,elapsed_time]= test_kernel(transposeMatrix_stride<float>,row,col);
    float elapsed_time = 0.0f;
    auto matrix = generateMatrix<float>(row,col);
    Matrix<float>  result(matrix.row(),matrix.col());

    using VauleType = decltype(matrix)::value_type;
    size_t byteSize = matrix.row() * matrix.col() * sizeof(VauleType);

    VauleType* dev_idata;
    VauleType* dev_odata;

    CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&dev_idata),byteSize));
    CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&dev_odata),byteSize));
    CUDACHECK(cudaMemcpy(dev_idata,matrix.data_ptr(),byteSize,cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid(512);
    CudaTimer timer("kernel");
    timer.startTiming();
    transposeMatrix_stride<<<blocksPerGrid,threadsPerBlock>>>(dev_odata,dev_idata,matrix.row(),matrix.col());
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
    elapsed_time = timer.stopTiming();
    CUDACHECK(cudaMemcpy(result.data_ptr(),dev_odata,byteSize,cudaMemcpyDeviceToHost));
    CUDACHECK(cudaFree(dev_idata));
    CUDACHECK(cudaFree(dev_odata));
    if(result != matrix.transpose())
    {
        std::cout<<"transposeMatrix_stride  is Incorrect!"<<std::endl;
    }
    return elapsed_time;
}




#define LOOP_TEST(test_func,n,row,col,baseline_time) \
{\
    float elapsed_time = 0.0f;\
    for(int i=0;i<n;++i)\
    {\
        elapsed_time += test_func(row,col);\
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


void loop_test()
{
    const int N=2;
    //dummy test for warm up
    test_transpose_base();
    //std::vector<int> test_sizes = {32,64,128,256,512,1024,2048,4096,8192};
    std::vector<int> test_sizes{2048};
    std::cout<<"Loop test "<<N<<" times"<<std::endl;

    for(auto size : test_sizes)
    {
        std::cout<<"Matrix size: "<<size<<"x"<<size<<std::endl;
        size_t row = size;
        size_t col = size;
        float baseline = BASELINE_TEST(test_transpose_base,N,row,col);
        LOOP_TEST(test_copyMatrix, N,row,col,baseline);
        LOOP_TEST(test_cublasTransposeMatrix, N,row,col,baseline);
        LOOP_TEST(test_transpose_shared, N,row,col,baseline);
        //LOOP_TEST(test_transpose_stride, N,row,col,baseline);
        LOOP_TEST(test_transpose_shared_2, N,row,col,baseline);
        LOOP_TEST(test_transposeMatrix_shared_unroll, N, row, col, baseline);
        std::cout<<std::endl;
    }

    
}



int main()
{
    printGPUInfo();
    loop_test();
    return 0;
}
