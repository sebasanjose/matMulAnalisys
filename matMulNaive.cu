#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define WIDTH 2048  // Large matrix size
#define EPSILON 1e-4

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

// Function to verify the results
bool verifyResult(float *A, float *B, float *C, int Width) {
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            float expected = 0.0;
            for (int k = 0; k < Width; k++) {
                expected += A[i * Width + k] * B[k * Width + j];
            }
            if (fabs(C[i * Width + j] - expected) > EPSILON) {
                printf("Mismatch at (%d, %d): expected %0.4f, got %0.4f\n", i, j, expected, C[i * Width + j]);
                return false;
            }
        }
    }
    return true;
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

    // Verify result
    if (verifyResult(h_A, h_B, h_C, WIDTH)) {
        printf("Matrix multiplication is correct!\n");
    } else {
        printf("Matrix multiplication verification failed!\n");
    }

    // Print elapsed time
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