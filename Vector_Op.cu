#include <stdio.h>

__global__ void dotProduct(float* A, float* B, float* C, float* D, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = (A[i] - B[i]) * (C[i] + D[i]);
    }
}

int main() {
    int n = 10000;
    float* h_A, * h_B, * h_C, * h_D, * h_result;
    float* d_A, * d_B, * d_C, * d_D, * d_result;
    size_t bytes = n * sizeof(float);

    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);
    h_D = (float*)malloc(bytes);
    h_result = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
        h_C[i] = i * 3;
        h_D[i] = i * 4;
    }

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_D, bytes);
    cudaMalloc(&d_result, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    dotProduct << <gridSize, blockSize >> > (d_A, d_B, d_C, d_D, d_result, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(start, 0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_result);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_result);

    printf("Time took to run the actual kernel inside the GPU: %f ms\n", kernelTime);
    printf("Time it took to copy data in and out of the GPU: %f ms\n", kernelTime - totalTime);

    return 0;
}