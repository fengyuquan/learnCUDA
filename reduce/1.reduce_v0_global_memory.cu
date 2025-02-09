#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREAD_PER_BLOCK 256 // 这里定义每个block中的线程数量，同时每个block中处理数组的大小也是这个

__global__ void reduce0(float *d_input, float *d_out)
{
    // 确定索引起始点
    float *inputPtr = d_input + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    // reduce计算
    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (tid % (i * 2) == 0)
        {
            inputPtr[tid] += inputPtr[tid + i];
        }
        __syncthreads();
    }

    // 保存结果
    if (tid == 0)
    {
        d_out[blockIdx.x] = inputPtr[0];
    }
}

bool check(float *a, float *b, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (fabs(a[i] - b[i]) > 0.005)
        {
            printf("The ans is wrong!\n");
            return false;
        }
    }

    printf("It's all right!\n");
    return true;
}

int main(int argc, char const *argv[])
{
    // 定义常量
    constexpr int N = 32 * 1024 * 1024;
    constexpr int BLOCK_NUM = N / THREAD_PER_BLOCK;

    // 定义变量
    float *input, *d_input;
    float *out, *d_out;
    float *res;

    // 初始化CPU内存
    input = (float *)malloc(N * sizeof(float)); // 输入的数据内存大小
    for (int i = 0; i < N; i++)
    {
        input[i] = 1;
    }
    out = (float *)malloc(BLOCK_NUM * sizeof(float))     // 输出的数据内存大小
        res = (float *)malloc(BLOCK_NUM * sizeof(float)) // 输出的数据内存大小

        // 初始化GPU内存
        cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_out, BLOCK_NUM * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 调用核心函数
    reduce0<<<dim3(BLOCK_NUM, 1), dim3(THREAD_PER_BLOCK, 1)>>>(d_input, d_out);

    // 复制值到CPU
    cudaMemcpy(out, d_out, BLOCK_NUM * sizeof(float), cudaMemcpyDeviceToHost);

    // 计算CPU端的值
    for (int i = 0; i < BLOCK_NUM; i++)
    {
        float cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; j++)
        {
            cur += input[i * THREAD_PER_BLOCK + j];
        }
        res[i] = cur;
    }

    // 检查计算结果准确性
    check(out, res, BLOCK_NUM);
    return 0;
}
