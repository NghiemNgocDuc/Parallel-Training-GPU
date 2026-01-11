#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 1024
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000
#define BATCH_SIZE 32
#define EPOCHS 10
#define LEARNING_RATE 0.01

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error), error); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}


// Timing structure
typedef struct {
    double memory_transfers;  // H2D only (data at start of batch)
    double gpu_compute;       // Forward + Loss + Backward + Update (all GPU)
    double total_time;
} TimingStats;


typedef struct {
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    float *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;
    float *d_fc1_output, *d_fc2_output, *d_grad_hidden, *d_grad_output;

    // PERSISTENT BUFFERS - NO MORE MALLOC/FREE PER BATCH
    float *d_input_batch;
    int *d_labels;           // Labels on GPU for loss computation
    float *d_loss; 

    cublasHandle_t cublas_handle; // Per-sample loss for reduction
} NeuralNetworkCUDA;

void load_data(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen data"); exit(EXIT_FAILURE); }
    fread(data, sizeof(float), size, f);
    fclose(f);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen labels"); exit(EXIT_FAILURE); }
    fread(labels, sizeof(int), size, f);
    fclose(f);
}

void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

void initialize_weights_host(float *weights, int rows, int cols) {
    float scale = sqrtf(2.0f / rows);
    for (int i = 0; i < rows * cols; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

void initialize_bias_host(float *bias, int size) {
    memset(bias, 0, size * sizeof(float));
}

__global__ void bias_add_kernel(float *output, float *bias, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size) {
        int col = idx % output_size;
        output[idx] += bias[col];
    }
}

__global__ void relu_activation(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

