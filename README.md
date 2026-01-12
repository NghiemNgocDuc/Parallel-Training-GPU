**Training a Neural Network with different methods, utilizing CPU and GPU**


**v1.py – PyTorch Baseline**
This version uses PyTorch with GPU (CUDA) support and serves as the starting point of the project. It relies on high-level PyTorch building blocks such as linear layers, ReLU activations, and cross-entropy loss, so most of the complexity is handled automatically by the framework. Memory management and GPU acceleration are taken care of under the hood, and PyTorch applies many built-in optimizations (for example, via cuDNN). The main goal of this version is to provide a clear, correct reference implementation and a performance baseline that later versions can be compared against.

**v2.py – NumPy Implementation**
This version reimplements the same model using only NumPy on the CPU, without any deep learning frameworks or GPU acceleration. All forward and backward computations are written manually, including gradient calculations and weight updates. Weights are initialized using He initialization to keep training stable. This implementation is intended to be educational: it exposes the actual math behind neural networks and backpropagation, making it easier for beginners to understand what frameworks like PyTorch are doing behind the scenes.

**v3.c – C / CPU Implementation**
In this version, the model is implemented in pure C and runs entirely on the CPU. Memory allocation and deallocation are handled manually, and the code includes detailed timing measurements for each major operation. Matrix operations are written with CPU performance in mind. This step establishes a low-level CPU performance baseline and helps identify where computation time is spent, which is especially useful before moving to a GPU-based implementation.

**v4.cu – Naive CUDA Kernels**
This version is the first to move fully onto the GPU using CUDA C. Core operations such as matrix multiplication, ReLU, bias addition, and softmax are implemented using custom CUDA kernels. Data must be explicitly transferred between the host (CPU) and device (GPU), which makes memory management more visible. The focus here is not maximum performance, but learning how GPU kernels work and how neural network computations can be parallelized on the GPU.

**v5.cu – cuBLAS Optimized**
This final version uses NVIDIA’s cuBLAS library to achieve much higher performance on the GPU. Instead of custom matrix multiplication kernels, it relies on highly optimized cuBLAS routines such as SGEMM and SAXPY. Memory buffers are reused to reduce allocation overhead, and unnecessary synchronization between the CPU and GPU is minimized. This version represents a more production-quality implementation, showing how real-world high-performance GPU code is typically written.
