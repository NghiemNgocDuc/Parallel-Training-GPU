#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Timing structure to hold the time taken for each operation
typedef struct{
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
#define LEARNING_RATE 0.01
#define EPOCHS 10

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

// load batch of data
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Failed to open data file");
        exit(EXIT_FAILURE);
    }

    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        perror("Failed to read data");
        exit(EXIT_FAILURE);
    }
    fclose(file);
}

//load labels
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Failed to open labels file");
        exit(EXIT_FAILURE);
    }

    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        perror("Failed to read labels");
        exit(EXIT_FAILURE);
    }
    fclose(file);
}


// Set all bias values to zero
void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// optimal uniform He init for weights
void initialize_weights(float *weights, int input_size, int output_size) {
    float scale = sqrtf(2.0f / input_size);
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

// Normalize data 
void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

// Custom softmax function
void softmax(float *x, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        float max = x[b * size];
        for (int i = 1; i < size; i++) {
            if (x[b * size + i] > max) max = x[b * size + i];
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[b * size + i] = expf(x[b * size + i] - max);
            sum += x[b * size + i];
        }
        for (int i = 0; i < size; i++) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

// Custom ReLU forward activation function
void relu_forward(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

// Adding biases 
void bias_forward(float *x, float *bias, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            x[b * size + i] += bias[i];
        }
    }
}

// Modify cross_entropy_loss to work with batches
float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}

// Zero out gradients
void zero_grad(float *grad, int size) {
    memset(grad, 0, size * sizeof(float));
}

// ReLU backward
void relu_backward(float *grad, float *x, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] *= (x[i] > 0);
    }
}

// Bias backward
void bias_backward(float *grad_bias, float *grad, int batch_size, int size) {
    for (int i = 0; i < size; i++) {
        grad_bias[i] = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_bias[i] += grad[b * size + i];
        }
    }
}

// Normal no transposed matrix multiplication
void matmul_a_b(float *A, float *B, float *C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            C[m * N + n] = 0.0f;
            for (int k = 0; k < K; k++) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}

// Matrix multiplication with B transposed
void matmul_a_bt(float *A, float *B, float *C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            C[m * N + n] = 0.0f;
            for (int k = 0; k < K; k++) {
                C[m * N + n] += A[m * K + k] * B[n * K + k];
            }
        }
    }
}

// Matrix multiplication with A transposed
void matmul_at_b(float *A, float *B, float *C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            C[m * N + n] = 0.0f;
            for (int k = 0; k < K; k++) {
                C[m * N + n] += A[k * M + m] * B[k * N + n];
            }
        }
    }
}

// Forward pass with detailed timing
void forward_pass(NeuralNetwork *nn, float *input, float *output, int batch_size, TimingStats *stats) {
    struct timespec start, end;

    // Layer 1: Input to Hidden
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_a_b(input, nn->weights1, output, batch_size, HIDDEN_SIZE, INPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul1 += get_time_diff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward(output, nn->bias1, batch_size, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias1 += get_time_diff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    relu_forward(output, batch_size * HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_relu += get_time_diff(start, end);

    // Layer 2: Hidden to Output
    float *hidden_output = (float *)malloc(batch_size * HIDDEN_SIZE * sizeof(float));
    memcpy(hidden_output, output, batch_size * HIDDEN_SIZE * sizeof(float));

    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_a_b(hidden_output, nn->weights2, output, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul2 += get_time_diff(start, end);
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward(output, nn->bias2, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias2 += get_time_diff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    softmax(output, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_softmax += get_time_diff(start, end);
}

// Compute output layer gradient
void compute_output_gradients(float *output, int *labels, float *grad_output, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }

    // Divide gradients by batch size
    for (int i = 0; i < batch_size * OUTPUT_SIZE; i++) {
        grad_output[i] /= batch_size;
    }
}

// Update weights and biases
void update_gradients(float *grad_weights, float *grad_bias, float *grad_layer, float *prev_layer, int batch_size, int prev_size, int curr_size) {
    for (int i = 0; i < curr_size; i++) {
        for (int j = 0; j < prev_size; j++) {
            for (int b = 0; b < batch_size; b++) {
                grad_weights[i * prev_size + j] += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
            }
            for (int b = 0; b < batch_size; b++) {
                grad_bias[i] += grad_layer[b * curr_size + i];
            }
        }
    }
}
// Backward pass with detailed timing
void backward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size, TimingStats *stats) {
    struct timespec start, end;

    // Set gradients to zero
    zero_grad(nn->grad_weights1, INPUT_SIZE * HIDDEN_SIZE);
    zero_grad(nn->grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE);
    zero_grad(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad(nn->grad_bias2, OUTPUT_SIZE);

    // Output layer gradient
    clock_gettime(CLOCK_MONOTONIC, &start);
    float *grad_output = (float *)malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    compute_output_gradients(grad_output, output, labels, batch_size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_output_grad += get_time_diff(start, end);

    // Gradients for weights2 and bias2
    







