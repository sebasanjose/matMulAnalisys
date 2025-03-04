#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 2048  // Large matrix size
#define TILE_WIDTH 16
#define EPSILON 1e-4
#define CHECK_COUNT 10

// Optimized matrix multiplication using tiling
__global__ void matMulTiled(float *A, float *B, float *C, int Width) {
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0.0;

    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        if (Row < Width && (ph * TILE_WIDTH + threadIdx.x) < Width)
            A_shared[threadIdx.y][threadIdx.x] = A[Row * Width + ph * TILE_WIDTH + threadIdx.x];
        else
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;

        if ((ph * TILE_WIDTH + threadIdx.y) < Width && Col < Width)
            B_shared[threadIdx.y][threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * Width + Col];
        else
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (Row < Width && Col < Width)
        C[Row * Width + Col] = Pvalue;
}

// Function to verify the results with random sampling
bool verifyResult(float *A, float *B, float *C, int Width) {
    srand(time(NULL));
    for (int i = 0; i < CHECK_COUNT; i++) {
        int row = rand() % Width;
        int col = rand() % Width;
        float expected = 0.0;
        for (int k = 0; k < Width; k++) {
            expected += A[row * Width + k] * B[k * Width + col];
        }
        if (fabs(C[row * Width + col] - expected) > EPSILON) {
            printf("Mismatch at (%d, %d): expected %0.4f, got %0.4f\n", row, col, expected, C[row * Width + col]);
            return false;
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
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (WIDTH + blockSize.y - 1) / blockSize.y);

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Execute the tiled kernel
    matMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, WIDTH);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result with random sampling
    if (verifyResult(h_A, h_B, h_C, WIDTH)) {
        printf("Matrix multiplication is correct!\n");
    } else {
        printf("Matrix multiplication verification failed!\n");
    }

    // Print elapsed time
    printf("\nElapsed Time for tiling case: %f ms\n", elapsedTime);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
