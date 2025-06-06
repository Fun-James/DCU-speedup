#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

// 编译
// hipcc sourcefile_dcu_optimized.cpp -o outputfile_dcu_optimized -O3
// 执行
// ./outputfile_dcu_optimized

#define N 1024
#define M 2048
#define P 512

// 定义块大小
#define TILE_SIZE 16

// 最优版本：向量化加载 + 避免Bank冲突
__global__ void matmul_kernel_optimized(const double* A, const double* B, double* C, int n, int m, int p) {
    __shared__ double As[TILE_SIZE][TILE_SIZE + 1];  // +1避免bank conflicts
    __shared__ double Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    double sum = 0.0;
    
    for (int t = 0; t < (m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 合并访问加载数据
        if (row < n && (t * TILE_SIZE + tx) < m) {
            As[ty][tx] = A[row * m + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        
        if ((t * TILE_SIZE + ty) < m && col < p) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * p + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // 展开内层循环提高计算效率
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < p) {
        C[row * p + col] = sum;
    }
}

void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (auto& x : mat) {
        x = dist(gen);
    }
}

void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

bool validate(const std::vector<double>& ref, const std::vector<double>& test, const std::string& version_name) {
    const double tolerance = 1e-6;
    int error_count = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - test[i]) > tolerance) {
            if (error_count < 5) {  // 只显示前5个错误
                std::cout << version_name << " validation failed at index " << i << ": " 
                          << ref[i] << " vs " << test[i] << " (diff: " << std::abs(ref[i] - test[i]) << ")" << std::endl;
            }
            error_count++;
        }
    }
    
    if (error_count > 0) {
        std::cout << version_name << " total errors: " << error_count << std::endl;
        return false;
    }
    return true;
}

double benchmark_kernel(void (*kernel)(const double*, const double*, double*, int, int, int),
                       double* d_A, double* d_B, double* d_C, 
                       std::vector<double>& C, dim3 gridSize, dim3 blockSize,
                       const std::string& kernel_name, int iterations = 10) {
    
    std::cout << "\n=== Testing " << kernel_name << " ===" << std::endl;
    
    // 预热
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, P);
    hipDeviceSynchronize();
    
    // 多次测试取平均值
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, P);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time = duration.count() / (double)iterations / 1000.0;  // 转换为毫秒
    
    // 复制结果回主机进行验证
    hipMemcpy(C.data(), d_C, N * P * sizeof(double), hipMemcpyDeviceToHost);
    
    std::cout << kernel_name << " average execution time: " << avg_time << " ms" << std::endl;
    
    return avg_time;
}

int main() {
    std::cout << "Optimized Matrix Multiplication using DCU/HIP" << std::endl;
    std::cout << "Matrix sizes: A(" << N << "x" << M << ") * B(" << M << "x" << P << ") = C(" << N << "x" << P << ")" << std::endl;
    
    // 检查设备信息
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    
    // 初始化矩阵
    std::cout << "\nInitializing matrices..." << std::endl;
    init_matrix(A);
    init_matrix(B);

    // CPU baseline计算
    std::cout << "Computing CPU baseline..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C_ref);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    std::cout << "CPU execution time: " << cpu_duration.count() << " milliseconds" << std::endl;

    // 分配设备内存
    double *d_A, *d_B, *d_C;
    hipMalloc(&d_A, N * M * sizeof(double));
    hipMalloc(&d_B, M * P * sizeof(double));
    hipMalloc(&d_C, N * P * sizeof(double));
    
    // 复制数据到设备
    hipMemcpy(d_A, A.data(), N * M * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), M * P * sizeof(double), hipMemcpyHostToDevice);
    
    // 设置网格和块大小
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((P + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    std::cout << "\nGrid size: (" << gridSize.x << ", " << gridSize.y << ")" << std::endl;
    std::cout << "Block size: (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
      // 测试最优版本
    double gpu_time = benchmark_kernel(matmul_kernel_optimized, d_A, d_B, d_C, C, gridSize, blockSize, "Optimized DCU Kernel");
    if (validate(C_ref, C, "Optimized")) {
        std::cout << "Optimized validation: PASSED" << std::endl;
    }
    
    // 性能总结
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "CPU baseline: " << cpu_duration.count() << " ms" << std::endl;
    
    double speedup_vs_cpu = (double)cpu_duration.count() / gpu_time;
    std::cout << "Optimized DCU Kernel: " << gpu_time << " ms (Speedup vs CPU: " << speedup_vs_cpu << "x)" << std::endl;
    

    // 释放设备内存
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    

    return 0;
}
