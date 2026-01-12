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

__global__ void relu_gradient(float *grad_output, float *fc1_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_output[idx] *= (fc1_output[idx] > 0) ? 1.0f : 0.0f;
    }
}

__global__ void bias_backward_kernel(float *grad_output, float *grad_bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        atomicAdd(&grad_bias[bias_idx], grad_output[idx]); // Use this since multiple threads can update the same bias, which not right due to racing for memory


    }


}

// All other funcs on GPU, not need to D2H and then gradient H2D again

__global__ void softmax_cross_entropy_backward_kernel(
    float *logits,
    int *labels,
    float *grad_output,
    int batch_size,
    int num_classes,
    float *loss_per_sample,


) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float shared[];
    float *sample_logits = shared;

    int tid = threadIdx.x;

    // Load logits into shared memory
    if (tid < num_classes) {
        sample_logits[tid] = logits[b * num_classes + tid];
    }
    __syncthreads();

    __shared__ float max_logit;
    __shared__ float sum_exp;

    if (tid == 0) {
        max_logit = -INFINITY;
        for (int i = 0; i < num_classes; i++) {
            if (sample_logits[i] > max_logit) {
                max_logit = sample_logits[i];
            }
        }
    }
    __syncthreads();

    if (tid < num_classes) {
        sample_logits[tid] = expf(sample_logits[tid] - max_logit);
    }
    __syncthreads();

    // Compute sum of exponentials
    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += sample_logits[i];
        }
    }
    __syncthreads();

    
}
