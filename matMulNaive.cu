#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 2048  // Large matrix size

// Naïve matrix multiplication (No tiling, inefficient)
__global__ void matMulNaive(float *A, float *B, float *C, int Width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < Width && Col < Width) {
        float Pvalue = 0.0;
        for (int k = 0; k < Width; k++) {
            Pvalue += A[Row * Width + k] * B[k * Width + Col];
        }
        C[Row * Width + Col] = Pvalue;
    }
}

int main() {
    size_t size = WIDTH * WIDTH * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_A[i] = static_cast<float>(rand() % 10);
        h_B[i] = static_cast<float>(rand() % 10);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (WIDTH + blockSize.y - 1) / blockSize.y);

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Execute the naive kernel
    matMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, WIDTH);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print input matrices and output matrix
    printf("Matrix A:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%0.2f ", h_A[i * WIDTH + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%0.2f ", h_B[i * WIDTH + j]);
        }
        printf("\n");
    }

    printf("\nMatrix C (Result):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%0.2f ", h_C[i * WIDTH + j]);
        }
        printf("\n");
    }

    printf("\nElapsed Time: %f ms\n", elapsedTime);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}