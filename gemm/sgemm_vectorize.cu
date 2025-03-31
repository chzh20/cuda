#include"cuda_runtime.h"
#include"device_launch_parameters.h"


template<typename T>
__global__ void segmm_vectorize(int m, int k, int n, T* A, T* B,T* C, T alpha, T betla)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int threadsPerBlock = (BM * BN) /(TM * TN);

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    //locate data in A and B
    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;


    //reshape thread idx 
    int threadRow = threadIdx.x /(BN/TN);
    int threadCol = threadIdx.x % (BN/TN);
    
    // load 128 bits/32 = 4 elements per thread at each iteration
    // threads in one block will load 256*4 = 1024 elements while there are BM*BK = 128*8 = 1024 elements in A
    // so we can load all elements in A into shared memory in one iteration
    int innerRowA = threadIdx.x / (BK/4);
    int innerColA = threadIdx.x % (BK/4);
    

    int innerRowB = threadIdx.x / (BN/4);
    int innerColB = threadIdx.x % (BN/4);

    T threadResult[TM *TN] = {0.0f};
    T regM[TM] = {0.0f};
    T regN[TN] = {0.0f};


    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    for(int bkIdx = 0; bkIdx < k; bkIdx += BK)
    {
        // load A and B into shared memory
        //[0] 的本质是 解引用操作，目的是从转换后的 float4* 指针中提取出具体的 float4 值。
        //若省略 [0]，则仅得到指针地址，而非实际数据。
        float4 temp = reinterpret_cast<float4*>(&A[innerRowA * k + innerColA*4])[0];
        As[(innerColA*4 + 0)*BM + innerRowA] = temp.x;
        As[(innerColA*4 + 1)*BM + innerRowA] = temp.y;
        As[(innerColA*4 + 2)*BM + innerRowA] = temp.z;
        As[(innerColA*4 + 3)*BM + innerRowA] = temp.w;

        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB*4])[0] = reinterpret_cast<float4*>(&B[innerRowB * n + innerColB*4])[0];

        __syncthreads();
        A+= BK;
        B+= BK * n;
        for(int dotIdx = 0; dotIdx < BK; dotIdx++)
        {
            for(int i = 0; i < TM; i++)
            {  //before transform: regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
            for(int i = 0; i < TN; i++)
            {
                regN[i] = Bs[dotIdx*BN + threadCol * TN + i];
            }
            for(int i = 0; i < TM; i++)
            {
                for(int j = 0; j < TN; j++)
                {
                    threadResult[i*TN+j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < TM; i++)
    {
        for(int j = 0; j < TN; j+=4)
        {
            float4 tmp = reinterpret_cast<float4*>(&C[(threadRow * TM + i) * n + threadCol * TN + j])[0];
            tmp.x = alpha * threadResult[i*TN+j] + betla * tmp.x;
            tmp.y = alpha * threadResult[i*TN+j+1] + betla * tmp.y;
            tmp.z = alpha * threadResult[i*TN+j+2] + betla * tmp.z;
            tmp.w = alpha * threadResult[i*TN+j+3] + betla * tmp.w;
            reinterpret_cast<float4*>(&C[(threadRow * TM + i) * n + threadCol * TN + j])[0] = tmp;
        }
    }


}
