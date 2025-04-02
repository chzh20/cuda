#include"utility.cuh"
#include "sgemm_naive.cuh"
#include "sgemm_shared.cuh"
#include <array>
#include<random>
#include"cublas_v2.h"
#include<iostream>
#include<cmath>
#include<iomanip>
#include<array>
#include <sys/types.h>
#include"sgemm_1D_blocktiling.cuh"
#include"sgemm_2D_blocktiling.cuh"
#include"sgemm_warptilling.cuh"
#include<fstream>
#include<sstream>
#include"Logger.h"


template <typename T>
T* gneraterandomMatrix(uint m, uint n, uint seed = 0)
{
    T *A = new T[m * n];
    std::random_device rd;
    std::mt19937 gen(seed != 0 ? seed : rd());
    std::uniform_int_distribution<int> dis(0, 100);
    for (uint i = 0; i < m * n; i++)
    {
        A[i] = static_cast<T>(dis(gen));
    }
    return A;
}

template<typename T>
bool checkResut(const T* A, const T* B, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (std::abs(A[i] - B[i]) > 1.0e-2)
        {
            std::cout <<std::fixed<<std::setprecision(3)<< "A[" << i << "]=" << A[i] << " B[" << i << "]=" << B[i] << std::endl;
            return false;
        }
    }
    return true;
}
template<typename T>
void printMatrix(const T* A, int m,int n)
{
   for(int i =0; i< m; ++i)
   {
        for(int j =0; j<n; j++)
        {
            std::cout<<A[i*n+j]<<" ";
        }
        std::cout<<std::endl;
   }
}

template<typename T>
void printMatrixAddress(const T* A, int m,int n)
{
   for(int i =0; i< m; ++i)
   {
        for(int j =0; j<n; j++)
        {
            std::cout<<&A[i*n+j]<<" ";
        }
        std::cout<<std::endl;
   }
}


template<typename T>
void transposeMatrix(const T* A, int m, int n, T* B)
{
    for(int i =0; i< m; ++i)
    {
        for(int j =0; j<n; j++)
        {
            B[j*m+i] = A[i*n+j];
        }
    }
}

template<typename T>
void cpu_test(int m, int n, int k, const T* A, T alpha, const T* B, T beta, T* C)
{
    // Initialize C to zero if beta is zero
    if (beta == 0.0f)
    {
        for (int i = 0; i < m * n; i++)
        {
            C[i] = 0;
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            T result = 0;
            for (int l = 0; l < k; l++)
            {
                result += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = alpha * result + beta * C[i * n + j];
        }
    }
}

template <typename T>
void cublas_test(int m, int n,int k, const T* A, T alpha, const T* B, T beta, T* C)
{
    T *d_A, *d_B, *d_C;

    CUDACHECK(cudaMalloc(&d_A, m * k * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_B, k * n * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_C, m * n * sizeof(T)));

    CUDACHECK(cudaMemcpy(d_A, A, m * k * sizeof(T), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, B, k * n * sizeof(T), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_C, C, m * n * sizeof(T), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);

    CudaTimer timer("cublas_sgemm");
    timer.startTiming();
    // C = alpha * A * B + beta * C and A is m x k, B is k x n, C is m x n
    //note that cublas use column major, so we need to transpose A and B
    //and the result is also in column major
    //cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, arow, bcol, acol, &alpha, d_A, acol, d_B, bcol, &beta, d_C, crow);
    
    
    
    
    // C= A*B  in row major ==> t = B^T * A^T = (B*A)^T in column major ==> C^T = B*A in column major
    // we will get C in row major
    // C^T(n,m) = B(n,k)*A(k,m) in column major;
    int crow = n;
    int ccol = m;
    int brow = n;
    int bcol = k;
    int arow = k;
    cublasSgemm(
        handle, 
        CUBLAS_OP_N, 
        CUBLAS_OP_N, 
        crow, // row of C^T or B^T
        ccol, // col of C^T or A^T
        bcol, // col of B^T or row of A^T
        &alpha, 
        d_B, 
        brow, //leading dimension of B^T,in column major, it is row of B^T
        d_A,  
        arow, //leading dimension of A^T,in column major, it is row of A^T
        &beta, 
        d_C, 
        crow); //leading dimension of C^T,in column major, it is row of C^T
    //CUDACHECK(cudaDeviceSynchronize());
    T milliseconds = timer.stopTiming();
    float Gflops = (2.0 * m * n * k* 1.e-9)/ (milliseconds / 1.0e3);
    
    std::string log = "cublas_sgemm: elapsed time: " + std::to_string(milliseconds) + " ms " + " GFLOPS: " + std::to_string(Gflops);
    Logger::getInstance().log(log);
    cublasDestroy(handle);

    CUDACHECK(cudaMemcpy(C, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaFree(d_A));
    CUDACHECK(cudaFree(d_B));
    CUDACHECK(cudaFree(d_C));
}

template<typename T>
struct KernelConfig
{  
    KernelConfig(uint m, uint n, uint k, T alpha, T beta, dim3 block_size, dim3 grid_size, std::string kernelname, T* A, T* B)
    {
        this->m = m;
        this->n = n;
        this->k = k;
        this->alpha = alpha;
        this->beta = beta;
        this->block_size = block_size;
        this->grid_size = grid_size;
        this->kernelname = kernelname;
        this->A = A;
        this->B = B;
    }
    uint m;
    uint n;
    uint k;
    T alpha;
    T beta;
    dim3 block_size;
    dim3 grid_size;
    std::string kernelname;
    T* A = nullptr;
    T* B = nullptr;
    T* C = nullptr;
    bool  cudaSharedmemCarveoutMaxShared = false;
};

#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))


template<typename KernelFunc, typename T,class Config= KernelConfig<T> >
void kernelTest(KernelFunc kernel,const char *kernelname, Config &config)
{
    
    T *d_A, *d_B, *d_C;
    uint m = config.m;
    uint n = config.n;
    uint k = config.k;
    T alpha = config.alpha;
    T beta = config.beta;
    T *A = config.A;
    T *B = config.B;
    T *C = config.C;
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    

    CUDACHECK(cudaMalloc(&d_A, m * k * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_B, k * n * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_C, m * n * sizeof(T)));

    CUDACHECK(cudaMemcpy(d_A, A, m * k * sizeof(T), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, B, k * n * sizeof(T), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_C, C, m * n * sizeof(T), cudaMemcpyHostToDevice));

    dim3 block_size = config.block_size;
    dim3 grid_size =  config.grid_size;

    if(config.cudaSharedmemCarveoutMaxShared)
    {
       // Set the shared memory carveout preference for the kernel to maximize shared memory
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    }
    
    CudaTimer timer(kernelname);
    timer.startTiming();
    kernel<<<grid_size, block_size >>> (m, n, k, alpha, d_A, d_B, beta, d_C);
    CUDACHECK(cudaGetLastError());
    cudaDeviceSynchronize(); 
    float milliseconds = timer.stopTiming();

    CUDACHECK(cudaMemcpy(config.C, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost));
    // Calculate the number of floating-point operations per second (GFLOPS)
    float Gflops = (2.0 * m * n * k * 1.e-9) / (milliseconds / 1.0e3);
    std::string log =std::string(kernelname) + ": elapsed time: " + std::to_string(milliseconds) + " ms " + " GFLOPS: " + std::to_string(Gflops);
    Logger::getInstance().log(log);
    CUDACHECK(cudaFree(d_A));
    CUDACHECK(cudaFree(d_B));
    CUDACHECK(cudaFree(d_C));
}

#define CHECKRESULT(base, target, targetname,size) \
    do { \
        if(!checkResut(base, target, size)) \
        { \
            std::cout<<targetname << " results are not the same !" << std::endl; \
        } \
    } while(0)


template<typename Type>
void test(uint m, uint n, uint k, Type alpha, Type beta)
{
    Type *A = gneraterandomMatrix<Type>(m, k,1);
    Type *B = gneraterandomMatrix<Type>(k, n,2);
    Type *C_cublas = new Type[m * n]{0.0f};
    Type *C_cpu = new Type[m * n]{0.0f};

    using  KernelType = void(*)(int, int, int, Type, const Type*, const Type*, Type, Type*);
    auto run_kernel=[&](KernelType kernel,const char *kernel_name, KernelConfig<Type> &config)
    {
        Type *out_put = new Type[config.m * config.n]{0};
        config.C = out_put;
        assert(config.A != nullptr);
        assert(config.B != nullptr);
        assert(config.C != nullptr);
        kernelTest<KernelType,Type>(kernel,kernel_name,config);
        delete[] out_put;
    };
    // cublas test
    cublas_test<Type>(m, n, k, A, alpha, B, beta, C_cublas);


    KernelConfig<Type> config_naive  {m, n, k, alpha, beta, 
                                  dim3(32,32),  //block_size, each thread is responsible for TM elements
                                  dim3(CEIL_DIV(m,32),CEIL_DIV(n,32)), //grid_size
                                "sgemm_naive", A, B};

    run_kernel(sgemm_naive,"sgemm_naive",config_naive);

    KernelConfig<Type> config_naive1  {m, n, k, alpha, beta, 
        dim3(32,32),  //block_size, each thread is responsible for TM elements
        dim3(CEIL_DIV(m,32),CEIL_DIV(n,32)), //grid_size
      "sgemm_naive_tans", A, B};

    run_kernel(sgemm_naive_tans,"sgemm_naive_tans",config_naive1);

    KernelConfig<Type> config_naive2 {m, n, k, alpha, beta, 
        dim3(32,32),  //block_size, each thread is responsible for TM elements
        dim3(CEIL_DIV(m,32),CEIL_DIV(n,32)), //grid_size
      "sgemm_coalescing", A, B};

     run_kernel(sgemm_coalescing,"sgemm_coalescing",config_naive2);



    KernelConfig<Type> config1 {m, n, k, alpha, beta, 
                                  dim3((BM*BN)/TM),  //block_size, each thread is responsible for TM elements
                                  dim3(CEIL_DIV(m,BM),CEIL_DIV(n,BN)), //grid_size
                                "sgemm_1D_Blocktiling", A, B};
    config1.cudaSharedmemCarveoutMaxShared = true;
    run_kernel(sgemm_1D_Blocktiling,"sgemm_1D_Blocktiling",config1);


    KernelConfig<Type> config2  {m, n, k, alpha, beta, 
                                  dim3(32*32),  //block_size, each thread is responsible for TM elements
                                  dim3(CEIL_DIV(m,32),CEIL_DIV(n,32)), //grid_size
                                "sgemm_shared2", A, B};
    //config2.cudaSharedmemCarveoutMaxShared = true;
    run_kernel(sgemm_shared2,"sgemm_shared2",config2);
    
    const int BM_2D = 128;
    const int BN_2D = 128;
    const int TM_2D = 8;
    const int TN_2D = 8;
    KernelConfig<Type> config3  {m, n, k, alpha, beta, 
                                  dim3(BM_2D*BN_2D /(TM_2D*TN_2D)),  //block_size, each thread is responsible for TM*TN elements
                                  dim3(CEIL_DIV(m,BM_2D),CEIL_DIV(n,BN_2D)), //grid_size
                                "sgemm_2D_Blocktiling", A, B};
    config3.cudaSharedmemCarveoutMaxShared = true;
    run_kernel(sgemm_2D_Blocktiling,"sgemm_2D_Blocktiling",config3);


    KernelConfig<Type> config4  {m, n, k, alpha, beta, 
                                  dim3(128),  //block_size, each thread is responsible for TM*TN elements
                                  dim3(CEIL_DIV(m,128),CEIL_DIV(n,128)), //grid_size
                                "sgemm_Warptiling", A, B};
    config4.cudaSharedmemCarveoutMaxShared = true;
    run_kernel(sgemm_Warptiling,"sgemm_Warptiling",config4);


    cublas_test<Type>(m, n, k, A, alpha, B, beta, C_cublas);
    //CHECKRESULT(C_cpu, C_cublas,"cublas", m * n);
    delete[] A;
    delete[] B;
    delete[] C_cublas;
    delete[] C_cpu;
}

void loopTest()
{
    constexpr int num = 8;
    std::array<uint,num> m = { 32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::array<uint,num> n = { 32, 64, 128, 256, 512, 1024, 2048,4096};
    std::array<uint,num> k = { 32, 64, 128, 256, 512, 1024, 2048,4096};
    float alpha = {1.0f};
    float beta = {1.0f};
    for(int i =7;i< num; i++)
    for(int i =7;i< num; i++)
    {
       //printf("m=%d, n=%d, k=%d\n", m[i], n[i], k[i]);
       std::string log = "Matrix Size: m=" + std::to_string(m[i]) + ", n=" + std::to_string(n[i]) + ", k=" + std::to_string(k[i]);
       Logger::getInstance().log(log);
       test(m[i], n[i], k[i], alpha, beta);
    }
    //test(64, 64, 64, alpha, beta);
}
int main()
{
    //CUDACHECK(cudaSetDevice(1));
    Logger::getInstance().setFileName("sgemm_log.txt");
    printGPUInfo();
    loopTest();
    return 0;
}


