#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define BATCH 1024
#define I 10
#define H 20
#define O 5

// CPU矩阵乘法函数
void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, 
                std::vector<double>& C, int M, int N, int K) {
    // A: M x K, B: K x N, C: M x N
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// CPU ReLU激活函数
void relu_cpu(std::vector<double>& A) {
    for (auto& val : A) {
        val = std::max(0.0, val);
    }
}

// CPU偏置添加函数
void add_bias_cpu(std::vector<double>& A, const std::vector<double>& bias, int M, int N) {
    // A: M x N matrix, bias: 1 x N vector
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            A[m * N + n] += bias[n];
        }
    }
}

// 随机初始化函数
void random_init(std::vector<double>& mat) {
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
    }
}

int main() {
    std::cout << "MLP Forward Propagation with CPU" << std::endl;
    std::cout << "Network Configuration: " << BATCH << "x" << I << " -> " << H << " -> " << O << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 初始化所有矩阵和向量
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_H(BATCH * H), h_Y(BATCH * O);

    // 随机初始化数据
    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    std::cout << "Data initialized successfully." << std::endl;
    
    auto compute_start = std::chrono::high_resolution_clock::now();

    // 隐藏层计算: H = X * W1
    std::cout << "Computing hidden layer..." << std::endl;
    matmul_cpu(h_X, h_W1, h_H, BATCH, H, I);
    
    // 添加偏置
    std::cout << "Adding bias..." << std::endl;
    add_bias_cpu(h_H, h_B1, BATCH, H);
    
    // 应用ReLU激活函数
    std::cout << "Applying ReLU..." << std::endl;
    relu_cpu(h_H);

    // 输出层计算: Y = H * W2
    std::cout << "Computing output layer..." << std::endl;
    matmul_cpu(h_H, h_W2, h_Y, BATCH, O, H);
    
    // 添加输出偏置
    std::cout << "Adding output bias..." << std::endl;
    add_bias_cpu(h_Y, h_B2, BATCH, O);

    auto compute_end = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_start);
   
    std::cout << "Computation completed successfully!" << std::endl;
    std::cout << "CPU computation time: " << compute_duration.count() << " microseconds" << std::endl;

    // 打印部分输出结果
    std::cout << "MLP Forward Propagation Results:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Sample " << i << " Output: ";
        for (int j = 0; j < O; ++j)
            std::cout << h_Y[i * O + j] << " ";
        std::cout << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Total execution time: " << total_duration.count() << " milliseconds" << std::endl;

    return 0;
}
