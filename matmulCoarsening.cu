#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

// Matrix dimensions and kernel parameters
#define N 2048                // 2048 x 2048 matrices
#define TILE_SIZE 16          // Block dimension
#define COARSENING_FACTOR 2   // Each thread computes 2 consecutive elements

// Coarsened matrix multiplication kernel
// Each thread computes COARSENING_FACTOR consecutive elements in one row of C.
__global__ void matMulCoarsened(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Each thread processes COARSENING_FACTOR columns starting at colStart
    int colStart = (blockIdx.x * blockDim.x + threadIdx.x) * COARSENING_FACTOR;

    if (row < N) {
        float sum[COARSENING_FACTOR] = {0.0f};
        // Loop over the common dimension of A and B
        for (int k = 0; k < N; k++) {
            float a = A[row * N + k];
            #pragma unroll
            for (int i = 0; i < COARSENING_FACTOR; i++) {
                int col = colStart + i;
                if (col < N)
                    sum[i] += a * B[k * N + col];
            }
        }
        // Write computed results back to C
        #pragma unroll
        for (int i = 0; i < COARSENING_FACTOR; i++) {
            int col = colStart + i;
            if (col < N)
                C[row * N + col] = sum[i];
        }
    }
}

int main() {
    // Calculate size in bytes for a matrix
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory for matrices A, B, and C
    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC = (float*)malloc(bytes);

    // Initialize matrices with random float values in [0,1]
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        hA[i] = ((float)rand()) / RAND_MAX;
        hB[i] = ((float)rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    // Copy host matrices A and B to device memory
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions.
    // Note: In the x-dimension, each thread computes COARSENING_FACTOR columns.
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(((N / COARSENING_FACTOR) + TILE_SIZE - 1) / TILE_SIZE,
                       (N + TILE_SIZE - 1) / TILE_SIZE);

    // Create CUDA events to measure kernel execution time.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event.
    cudaEventRecord(start, 0);

    // Launch the coarsened matrix multiplication kernel.
    matMulCoarsened<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);

    // Record the stop event.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds.
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f ms\n", elapsedTime);

    // Copy the result matrix C from device back to host.
    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    // Verify results for 10 random positions by comparing GPU results with CPU-computed values.
    printf("Verifying 10 random positions:\n");
    for (int i = 0; i < 10; i++) {
        int row = rand() % N;
        int col = rand() % N;
        // Compute the reference value on the CPU.
        float ref = 0.0f;
        for (int k = 0; k < N; k++) {
            ref += hA[row * N + k] * hB[k * N + col];
        }
        float gpuVal = hC[row * N + col];
        printf("Position (%d, %d): GPU = %f, CPU = %f, diff = %e\n",
               row, col, gpuVal, ref, fabs(gpuVal - ref));
    }

    // Clean up device and host memory.
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);

    // Destroy CUDA events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
