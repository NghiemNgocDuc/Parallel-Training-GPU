#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

typedef struct {
    double data_loading;
    double fwd_matmul1;
    double fwd_bias1;
    double fwd_relu;
    double fwd_matmul2;
    double fwd_bias2;
    double fwd_softmax;
    double cross_entropy;
    double bwd_output_grad;
    double bwd_matmul2;
    double bwd_bias2;
    double bwd_relu;
    double bwd_matmul1;
    double bwd_bias1;
    double weight_updates;
    double total_time;
} TimingStats;

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

#define INPUT_SIZE 784
#define HIDDEN_SIZE 1024
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 10
#define LEARNING_RATE 0.01

typedef struct {
    float *weights1;
    float *bias1;
    float *weights2;
    float *bias2;
    float *grad_weights1;
    float *grad_weights2;
    float *grad_bias1;
    float *grad_bias2;
} NeuralNetwork;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// load batched img data
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// optimal uniform He init for weights
void initialize_weights(float *weights, int input_size, int output_size) {
    float scale = sqrtf(2.0f / input_size);
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

// basic init for biases
void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// normalize data using MNIST mean and std
void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

// GPU part

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
// Matrix multiplication with A transposed
__global__ void matmul_at_b(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; k++) {
            value += A[k * M + row] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

//Matrix multiplication with B transposed
__global__ void matmul_a_bt(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = value;
    }
}

__global__ void relu_forward_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

// Add bias kernel
__global__ void bias_forward_kernel(float *x, float *bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;

    if (b < batch_size && i < size) {
        x[idx] += bias[i];
    }
}

// Zero gradients kernel
__global__ void zero_grad_kernel(float *grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

// Softmax kernel but runs on CPU for numerical stability
__global__void softmax_kernel(float *x, int batch_size, int size) {
    int idx = blockIdx.x;
    if (idx < batch_size) {
        float max_val = -INFINITY;
        for (int i = 0; i < size; i++) {
            if (x[idx * size + i] > max_val) {
                max_val = x[idx * size + i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[idx * size + i] = expf(x[idx * size + i] - max_val);
            sum += x[idx * size + i];
        }
        for (int i = 0; i < size; i++) {
            x[idx * size + i] /= sum;
        }
    }
}

// Compute output gradient kernel
__global__ void compute_output_gradient_kernel(float *predictions, int *labels, float *output_grad, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size){
        for (int i = 0; i < output_size; i++) {
            output_grad[idx * output_size + i] = predictions[idx * output_size + i];
        }
        int label = labels[idx];
        output_grad[idx * output_size + label] -= 1.0f;

        // Divide by batch size
        for (int i = 0; i < output_size; i++) {
            output_grad[idx * output_size + i] /= batch_size;
        }

    }
}

// Relu backward kernel
__global__ void relu_backward_kernel(float *grad, float *activations, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (activations[idx] <= 0.0f) {
            grad[idx] = 0.0f;
        }
    }
}

// Bias backward kernel
__global__ void bias_backward_kernel(float *grad_bias, float *grad, int batch_size, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad[b * size + i];
        }
        grad_bias[i] = sum;
    }
}

// Weight update kernel
__global__ void weight_update_kernel(float *weights, float *grad_weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
    }
}

// Cross entropy loss (CPU - same as v3.c)
float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}

// Modified forward function with timing (GPU version of v3.c)
void forward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    dim3 block_size(32, 32);

    // Input to Hidden Layer
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_size1((HIDDEN_SIZE + block_size.x - 1) / block_size.x,
                          (batch_size + block_size.y - 1) / block_size.y);
    matmul_kernel<<<grid_size1, block_size>>>(input, nn->weights1, hidden, batch_size, HIDDEN_SIZE, INPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul1 += get_time_diff(start, end);

    // Add bias1
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias1 += get_time_diff(start, end);

    // Apply ReLU
    clock_gettime(CLOCK_MONOTONIC, &start);
    relu_forward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_relu += get_time_diff(start, end);

    // Hidden to Output (Hidden @ W2)
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_size2((OUTPUT_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);
    matmul_a_b_kernel<<<grid_size2, block_size>>>(hidden, nn->weights2, output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul2 += get_time_diff(start, end);

    // Add bias2
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias2 += get_time_diff(start, end);

    // Apply softmax
    clock_gettime(CLOCK_MONOTONIC, &start);
    softmax_kernel<<<batch_size, 1>>>(output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_softmax += get_time_diff(start, end);
}

// Modified backward function with timing (GPU version of v3.c)
void backward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    dim3 block_size(32, 32);

    // Initialize gradients to zero
    zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256>>>(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad_kernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);

    // Allocate temp gradients on GPU
    float *grad_output, *dX2, *d_ReLU_out;
    CUDA_CHECK(cudaMalloc(&grad_output, batch_size * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dX2, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ReLU_out, batch_size * HIDDEN_SIZE * sizeof(float)));

    // Compute gradients for output layer
    clock_gettime(CLOCK_MONOTONIC, &start);
    compute_output_gradients_kernel<<<(batch_size + 255) / 256, 256>>>(grad_output, output, labels, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_output_grad += get_time_diff(start, end);

    / Update gradients for weights2 (W2.grad = hidden.T @ grad_output)
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_weights2((OUTPUT_SIZE + block_size.x - 1) / block_size.x, (HIDDEN_SIZE + block_size.y - 1) / block_size.y);
    matmul_at_b_kernel<<<grid_weights2, block_size>>>(hidden, grad_output, nn->grad_weights2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul2 += get_time_diff(start, end);

    // Update gradients for bias2
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->grad_bias2, grad_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias2 += get_time_diff(start, end);
