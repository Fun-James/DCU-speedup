#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>


// 编译
// hipcc sourcefile_dcu.cpp -o outputfile_dcu
// 执行
// ./outputfile_dcu

#define N 1024
#define M 2048
#define P 512

// HIP/DCU矩阵乘法核函数
__global__ void matmul_kernel(const double* A, const double* B, double* C, int n, int m, int p) {
    // 获取当前线程的全局索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查边界条件
    if (row < n && col < p) {
        double sum = 0.0;
        // 计算 C[row][col] = Σ(A[row][k] * B[k][col])
        for (int k = 0; k < m; ++k) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
    return;
}

void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);  // 固定种子以保证结果可重现
    std::uniform_real_distribution<double> dist(-100.0, 100.0);  // 使用更大的范围以便测试
    for (auto& x : mat) {
        x = dist(gen);
    }
    return;
}

void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    // CPU基准实现：标准三重循环矩阵乘法
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
    return;
}

bool validate(const std::vector<double>& ref, const std::vector<double>& test) {
    const double tolerance = 1e-6;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - test[i]) > tolerance) {
            std::cout << "Validation failed at index " << i << ": " 
                      << ref[i] << " vs " << test[i] << " (diff: " << std::abs(ref[i] - test[i]) << ")" << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "Matrix Multiplication using DCU/HIP" << std::endl;
    std::cout << "Matrix sizes: A(" << N << "x" << M << ") * B(" << M << "x" << P << ") = C(" << N << "x" << P << ")" << std::endl;
    
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    
    // 初始化矩阵
    std::cout << "Initializing matrices..." << std::endl;
    init_matrix(A);
    init_matrix(B);

    // CPU baseline计算
    std::cout << "Computing CPU baseline..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C_ref);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    std::cout << "CPU execution time: " << cpu_duration.count() << " milliseconds" << std::endl;

    // DCU/HIP计算
    std::cout << "Computing DCU/HIP version..." << std::endl;
    double *d_A, *d_B, *d_C;
    
    // 分配设备内存
    hipMalloc(&d_A, N * M * sizeof(double));
    hipMalloc(&d_B, M * P * sizeof(double));
    hipMalloc(&d_C, N * P * sizeof(double));
    
    // 复制数据到设备
    hipMemcpy(d_A, A.data(), N * M * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), M * P * sizeof(double), hipMemcpyHostToDevice);
    
    // 设置线程块和网格大小
    dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Grid size: (" << gridSize.x << ", " << gridSize.y << ")" << std::endl;
    std::cout << "Block size: (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    
    // 启动核函数并测量时间
    auto dcu_start = std::chrono::high_resolution_clock::now();
    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, P);
    hipDeviceSynchronize();  // 等待核函数完成
    auto dcu_end = std::chrono::high_resolution_clock::now();
    auto dcu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dcu_end - dcu_start);
    
    // 复制结果回主机
    hipMemcpy(C.data(), d_C, N * P * sizeof(double), hipMemcpyDeviceToHost);
    
    std::cout << "DCU execution time: " << dcu_duration.count() << " milliseconds" << std::endl;
    std::cout << "Speedup: " << (double)cpu_duration.count() / dcu_duration.count() << "x" << std::endl;
    
    // 验证结果
    if (validate(C_ref, C)) {
        std::cout << "[HIP] Validation: PASSED" << std::endl;
    } else {
        std::cout << "[HIP] Validation: FAILED" << std::endl;
    }
    
    // 释放设备内存
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    std::cout << "Matrix multiplication completed successfully!" << std::endl;
    // 需额外增加性能评测代码或其他工具进行评测
    return 0;
}
