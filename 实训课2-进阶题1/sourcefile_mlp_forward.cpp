#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

// 错误检查宏
#define HIP_CHECK(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define BATCH 2048
#define I 1024
#define H 512
#define O 128
#define BLOCK_SIZE 32  // 块大小，可以根据DCU特性调整

// 主要修改函数
__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    // A: M x K, B: K x N, C: M x N
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    return;
}

__global__ void relu_kernel(double* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmax(0.0, A[idx]);
    }
    return;
}

// 添加偏置向量的内核函数
__global__ void add_bias_kernel(double* A, const double* bias, int M, int N) {
    // A: M x N matrix, bias: 1 x N vector
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        A[idx] += bias[col];
    }
    return;
}

void random_init(std::vector<double>& mat) {
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
    }
    return;
}

int main() {
    std::cout << "MLP Forward Propagation with HIP" << std::endl;
    std::cout << "Network Configuration: " << BATCH << "x" << I << " -> " << H << " -> " << O << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_H(BATCH * H), h_Y(BATCH * O);

    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    std::cout << "Data initialized successfully." << std::endl;

    // 以下均为主要修改部分
    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    
    // 分配GPU内存
    HIP_CHECK(hipMalloc(&d_X, BATCH * I * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W1, I * H * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B1, H * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_H, BATCH * H * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W2, H * O * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B2, O * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Y, BATCH * O * sizeof(double)));
    
    std::cout << "GPU memory allocated successfully." << std::endl;
    
    // 将数据从主机复制到设备
    HIP_CHECK(hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice));
    
    std::cout << "Data copied to GPU successfully." << std::endl;
    
    auto compute_start = std::chrono::high_resolution_clock::now();

    // Hidden layer: H = X * W1
    std::cout << "Computing hidden layer..." << std::endl;
    dim3 blockSize1(16, 16);
    dim3 gridSize1((H + blockSize1.x - 1) / blockSize1.x, (BATCH + blockSize1.y - 1) / blockSize1.y);
    matmul_kernel<<<gridSize1, blockSize1>>>(d_X, d_W1, d_H, BATCH, H, I);
    hipDeviceSynchronize();

    // Add bias and apply ReLU
    int blockSize_bias = 256;
    int gridSize_bias = (BATCH * H + blockSize_bias - 1) / blockSize_bias;
    std::cout << "Adding bias and applying ReLU..." << std::endl;
    add_bias_kernel<<<gridSize_bias, blockSize_bias>>>(d_H, d_B1, BATCH, H);
    hipDeviceSynchronize();
    
    relu_kernel<<<gridSize_bias, blockSize_bias>>>(d_H, BATCH * H);
    hipDeviceSynchronize();

    // Output layer: Y = H * W2
    std::cout << "Computing output layer..." << std::endl;
    dim3 blockSize2(16, 16);
    dim3 gridSize2((O + blockSize2.x - 1) / blockSize2.x, (BATCH + blockSize2.y - 1) / blockSize2.y);
    matmul_kernel<<<gridSize2, blockSize2>>>(d_H, d_W2, d_Y, BATCH, O, H);
    hipDeviceSynchronize();

    // Add output bias
    int blockSize_out = 256;
    int gridSize_out = (BATCH * O + blockSize_out - 1) / blockSize_out;
    std::cout << "Adding output bias..." << std::endl;
    add_bias_kernel<<<gridSize_out, blockSize_out>>>(d_Y, d_B2, BATCH, O);
    hipDeviceSynchronize();

    // 将结果从设备复制回主机
    HIP_CHECK(hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost));
    
    auto compute_end = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_start);
   
    std::cout << "Computation completed successfully!" << std::endl;
    std::cout << "GPU computation time: " << compute_duration.count() << " microseconds" << std::endl;

    // Print a few output values
    std::cout << "MLP Forward Propagation Results:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Sample " << i << " Output: ";
        for (int j = 0; j < O; ++j)
            std::cout << h_Y[i * O + j] << " ";
        std::cout << std::endl;
    }

    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_W1));
    HIP_CHECK(hipFree(d_B1));
    HIP_CHECK(hipFree(d_H));
    HIP_CHECK(hipFree(d_W2));
    HIP_CHECK(hipFree(d_B2));
    HIP_CHECK(hipFree(d_Y));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "GPU memory freed successfully." << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << " milliseconds" << std::endl;

    return 0;
}
