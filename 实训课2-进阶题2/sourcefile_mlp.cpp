#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip> // 新增: 用于格式化输出

// 编译文件
// hipcc sourcefile_mlp.cpp -o mlp_full_dcu
// 执行文件
// ./mlp_full_dcu 或者 hipprof ./mlp_full_dcu

// 预定义参数，可根据需求修改 - 基于时序分析优化
#define INPUT_DIM 50       // 基于自相关分析的最佳窗口大小
#define HIDDEN_DIM 96      // 适当增加隐藏层以处理更多输入特征
#define OUTPUT_DIM 1
#define BATCH_SIZE 128     // 减小批次大小以适应更大的输入维度
#define EPOCHS 2000        // 增加训练轮次以充分学习时序模式
#define LEARNING_RATE 3e-4 // 调整学习率以适应新的网络规模
#define DROPOUT_RATE 0.12  // 轻微降低dropout，保留更多时序信息
#define L2_LAMBDA 2e-5     // 适当调整L2正则化     


// 以下函数和main函数均不为固定形式，可自行按照需求修改

// HIP kernels函数形式，需要自行设计
__global__ void matmul(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    return;
}

__global__ void compute_output_grad(const double* pred, const double* target, double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 2.0 * (pred[idx] - target[idx]) / size;  // MSE gradient
    }
    return;
}

__global__ void compute_relu_backward(double* delta, const double* activ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta[idx] = (activ[idx] > 0.0) ? delta[idx] : 0.0;
    }
    return;
}

__global__ void compute_mse_loss(const double* pred, const double* target, double* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double diff = pred[idx] - target[idx];
        loss[idx] = diff * diff;
    }
    return;
}

__global__ void sgd_update(double* weights, const double* grad, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
    return;
}

// L2正则化的SGD更新（包含权重衰减）
__global__ void sgd_update_l2(double* weights, const double* grad, double lr, double l2_lambda, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 梯度 = 原梯度 + L2正则化项
        // 权重更新 = 权重 - 学习率 * (梯度 + λ * 权重)
        weights[idx] -= lr * (grad[idx] + l2_lambda * weights[idx]);
    }
    return;
}

// 计算L2正则化损失
__global__ void compute_l2_loss(const double* weights, double* l2_loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        l2_loss[idx] = weights[idx] * weights[idx];
    }
    return;
}

// 转置矩阵乘法：A^T * B
__global__ void matmul_transpose_A(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < K && col < N) {
        double sum = 0.0;
        for (int m = 0; m < M; ++m) {
            sum += A[m * K + row] * B[m * N + col];  // A转置
        }
        C[row * N + col] = sum;
    }
    return;
}

// 矩阵乘法：A * B^T
__global__ void matmul_transpose_B(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        double sum = 0.0;
        for (int n = 0; n < N; ++n) {
            sum += A[row * N + n] * B[col * N + n];  // B转置
        }
        C[row * K + col] = sum;
    }
    return;
}

// 计算偏置梯度（对批次求和）
__global__ void compute_bias_grad(const double* grad, double* bias_grad, int batch_size, int feature_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < feature_size) {
        double sum = 0.0;
        for (int b = 0; b < batch_size; ++b) {
            sum += grad[b * feature_size + idx];
        }
        bias_grad[idx] = sum;
    }
    return;
}

// 简单的线性同余随机数生成器
__device__ double simple_random(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return (double)(*seed) / 0x7fffffff;
}

// Dropout前向传播 (训练模式)
__global__ void dropout_forward_train(double* data, double* mask, 
                                     double dropout_rate, int size, unsigned int epoch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 使用线程ID和epoch作为种子
        unsigned int seed = idx + epoch * size + blockIdx.x * 1000;
        double rand_val = simple_random(&seed);
        
        if (rand_val < dropout_rate) {
            mask[idx] = 0.0;
            data[idx] = 0.0;
        } else {
            mask[idx] = 1.0 / (1.0 - dropout_rate);  // 缩放因子
            data[idx] = data[idx] * mask[idx];
        }
    }
}

// Dropout反向传播
__global__ void dropout_backward(double* grad, const double* mask, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = grad[idx] * mask[idx];
    }
}

// 加载带宽数据
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return data;
    }
    
    std::string line;
    std::getline(file, line);
    
    // 去掉开头的 '[' 和结尾的 ']'
    if (line.front() == '[') line.erase(0, 1);
    if (line.back() == ']') line.pop_back();
    
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // 去掉空格
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) {
            data.push_back(std::stod(token));
        }
    }
    
    std::cout << "Loaded " << data.size() << " data points from " << filename << std::endl;
    return data;
}

// 创建数据集
void create_dataset(const std::vector<double>& data,
                    std::vector<double>& X,
                    std::vector<double>& y) {
    int window_size = INPUT_DIM;
    int num_samples = data.size() - window_size;
    
    X.resize(num_samples * window_size);
    y.resize(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        // 输入特征：窗口内的数据
        for (int j = 0; j < window_size; ++j) {
            X[i * window_size + j] = data[i + j];
        }
        // 目标值：下一个时间点的数据
        y[i] = data[i + window_size];
    }
    
    std::cout << "Created dataset with " << num_samples << " samples" << std::endl;
    return;
}

// 数据归一化处理
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
    return;
}

// 数据反归一化处理
void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
    return;
}

// ReLU激活函数
__global__ void relu_activation(double* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmax(0.0, data[idx]);
    }
    return;
}

// 添加偏置
__global__ void add_bias(double* data, const double* bias, int batch_size, int feature_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * feature_size;
    if (idx < total_size) {
        int feature_idx = idx % feature_size;
        data[idx] += bias[feature_idx];
    }
    return;
}

// 计算总损失
double compute_total_loss(const double* loss_array, int size) {
    double total = 0.0;
    for (int i = 0; i < size; ++i) {
        total += loss_array[i];
    }
    return total / size;
}

// 随机初始化权重
void init_weights(std::vector<double>& weights, int size) {
    weights.resize(size);
    for (int i = 0; i < size; ++i) {
        weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;  // [-0.05, 0.05]
    }
    return;
}

// 保存模型
void save_model(const std::vector<double>& w1, const std::vector<double>& b1,
                const std::vector<double>& w2, const std::vector<double>& b2,
                const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot save model to " << filename << std::endl;
        return;
    }
    
    // 保存网络结构信息
    file << INPUT_DIM << " " << HIDDEN_DIM << " " << OUTPUT_DIM << std::endl;
    
    // 保存权重和偏置
    for (double val : w1) file << val << " ";
    file << std::endl;
    for (double val : b1) file << val << " ";
    file << std::endl;
    for (double val : w2) file << val << " ";
    file << std::endl;
    for (double val : b2) file << val << " ";
    file << std::endl;
    
    file.close();
    std::cout << "Model saved to " << filename << std::endl;
}

// 保存预测结果到CSV文件
void save_predictions_to_csv(const std::vector<double>& actuals_norm,
                             const std::vector<double>& predictions_norm,
                             const std::vector<double>& actuals_real,
                             const std::vector<double>& predictions_real,
                             const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing predictions." << std::endl;
        return;
    }

    // 写入表头
    file << "Actual_Normalized,Predicted_Normalized,Actual_Real,Predicted_Real\n";

    // 写入数据
    for (size_t i = 0; i < actuals_norm.size(); ++i) {
        file << actuals_norm[i] << ","
             << predictions_norm[i] << ","
             << actuals_real[i] << ","
             << predictions_real[i] << "\n";
    }

    file.close();
    std::cout << "Predictions saved to " << filename << std::endl;
}

// ----------------------------- Main -------------------------------
int main() {
    // 设置随机种子
    srand(42);
    
    // 读取带宽json文件
    std::vector<double> raw_data = load_json_bandwidth("starlink_bw.json");
    if (raw_data.empty()) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }
    
    // 数据归一化
    double min_val, max_val;
    std::vector<double> normalized_data = raw_data;
    normalize_data(normalized_data, min_val, max_val);
    
    // 创建数据集
    std::vector<double> X, y;
    create_dataset(normalized_data, X, y);
    
    int num_samples = y.size();
    int train_size = (int)(num_samples * 0.8);
    int test_size = num_samples - train_size;
    
    std::cout << "Train samples: " << train_size << ", Test samples: " << test_size << std::endl;
    
    // 初始化网络权重和偏置
    std::vector<double> w1, b1, w2, b2;
    init_weights(w1, INPUT_DIM * HIDDEN_DIM);
    init_weights(b1, HIDDEN_DIM);
    init_weights(w2, HIDDEN_DIM * OUTPUT_DIM);
    init_weights(b2, OUTPUT_DIM);
    
    // 分配GPU内存
    double *d_X, *d_y, *d_w1, *d_b1, *d_w2, *d_b2;
    double *d_hidden, *d_output, *d_output_grad, *d_hidden_grad;
    double *d_w1_grad, *d_b1_grad, *d_w2_grad, *d_b2_grad;
    double *d_loss, *h_loss;
    double *d_dropout_mask;  // dropout掩码
    double *d_l2_loss, *h_l2_loss;  // L2正则化损失
    
    // 为批处理分配内存
    int max_batch = BATCH_SIZE;
    h_loss = new double[max_batch];
    h_l2_loss = new double[std::max(INPUT_DIM * HIDDEN_DIM, HIDDEN_DIM * OUTPUT_DIM)];
    
    hipMalloc(&d_X, max_batch * INPUT_DIM * sizeof(double));
    hipMalloc(&d_y, max_batch * sizeof(double));
    hipMalloc(&d_w1, INPUT_DIM * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_b1, HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_w2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_b2, OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_hidden, max_batch * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_output, max_batch * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_output_grad, max_batch * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_hidden_grad, max_batch * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_w1_grad, INPUT_DIM * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_b1_grad, HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_w2_grad, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_b2_grad, OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_loss, max_batch * sizeof(double));
    hipMalloc(&d_dropout_mask, max_batch * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_l2_loss, std::max(INPUT_DIM * HIDDEN_DIM, HIDDEN_DIM * OUTPUT_DIM) * sizeof(double));
    
    // 拷贝初始权重到GPU
    hipMemcpy(d_w1, w1.data(), w1.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b1, b1.data(), b1.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w2, w2.data(), w2.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b2, b2.data(), b2.size() * sizeof(double), hipMemcpyHostToDevice);
    
    // 训练MLP网络
    double total_training_time_ms = 0.0; // 新增: 总训练时间
    std::cout << std::fixed << std::setprecision(6); // 设置输出精度

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        double total_loss = 0.0;
        int num_batches = 0;
        
        for (int start_idx = 0; start_idx < train_size; start_idx += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, train_size - start_idx);
            num_batches++;
            
            // 准备批数据
            std::vector<double> batch_X(current_batch_size * INPUT_DIM);
            std::vector<double> batch_y(current_batch_size);
            
            for (int i = 0; i < current_batch_size; ++i) {
                for (int j = 0; j < INPUT_DIM; ++j) {
                    batch_X[i * INPUT_DIM + j] = X[(start_idx + i) * INPUT_DIM + j];
                }
                batch_y[i] = y[start_idx + i];
            }
            
            // 拷贝批数据到GPU
            hipMemcpy(d_X, batch_X.data(), current_batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(d_y, batch_y.data(), current_batch_size * sizeof(double), hipMemcpyHostToDevice);
            
            // 前向传播
            // 隐藏层：X * W1 + b1
            dim3 blockSize(16, 16);
            dim3 gridSize((HIDDEN_DIM + blockSize.x - 1) / blockSize.x, 
                         (current_batch_size + blockSize.y - 1) / blockSize.y);
            matmul<<<gridSize, blockSize>>>(d_X, d_w1, d_hidden, current_batch_size, HIDDEN_DIM, INPUT_DIM);
            
            // 添加偏置
            int threads = 256;
            int blocks = (current_batch_size * HIDDEN_DIM + threads - 1) / threads;
            add_bias<<<blocks, threads>>>(d_hidden, d_b1, current_batch_size, HIDDEN_DIM);
            
            // ReLU激活
            relu_activation<<<blocks, threads>>>(d_hidden, current_batch_size * HIDDEN_DIM);
            
            // 应用Dropout (训练模式)
            dropout_forward_train<<<blocks, threads>>>(d_hidden, d_dropout_mask, 
                                                      DROPOUT_RATE, current_batch_size * HIDDEN_DIM, epoch);
            
            // 输出层：hidden * W2 + b2
            gridSize = dim3((OUTPUT_DIM + blockSize.x - 1) / blockSize.x,
                           (current_batch_size + blockSize.y - 1) / blockSize.y);
            matmul<<<gridSize, blockSize>>>(d_hidden, d_w2, d_output, current_batch_size, OUTPUT_DIM, HIDDEN_DIM);
            
            blocks = (current_batch_size * OUTPUT_DIM + threads - 1) / threads;
            add_bias<<<blocks, threads>>>(d_output, d_b2, current_batch_size, OUTPUT_DIM);
            
            // 计算损失
            compute_mse_loss<<<blocks, threads>>>(d_output, d_y, d_loss, current_batch_size);
            hipMemcpy(h_loss, d_loss, current_batch_size * sizeof(double), hipMemcpyDeviceToHost);
            double mse_loss = compute_total_loss(h_loss, current_batch_size);
            
            // 计算L2正则化损失
            double l2_loss = 0.0;
            // W1的L2损失
            blocks = (INPUT_DIM * HIDDEN_DIM + threads - 1) / threads;
            compute_l2_loss<<<blocks, threads>>>(d_w1, d_l2_loss, INPUT_DIM * HIDDEN_DIM);
            hipMemcpy(h_l2_loss, d_l2_loss, INPUT_DIM * HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToHost);
            for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; ++i) {
                l2_loss += h_l2_loss[i];
            }
            
            // W2的L2损失
            blocks = (HIDDEN_DIM * OUTPUT_DIM + threads - 1) / threads;
            compute_l2_loss<<<blocks, threads>>>(d_w2, d_l2_loss, HIDDEN_DIM * OUTPUT_DIM);
            hipMemcpy(h_l2_loss, d_l2_loss, HIDDEN_DIM * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost);
            for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; ++i) {
                l2_loss += h_l2_loss[i];
            }
            
            // 总损失 = MSE损失 + λ * L2损失
            total_loss += mse_loss + L2_LAMBDA * l2_loss;
            
            // 反向传播
            // 输出层梯度
            compute_output_grad<<<blocks, threads>>>(d_output, d_y, d_output_grad, current_batch_size);
            
            // 计算W2梯度：hidden^T * output_grad
            gridSize = dim3((OUTPUT_DIM + blockSize.x - 1) / blockSize.x,
                           (HIDDEN_DIM + blockSize.y - 1) / blockSize.y);
            matmul_transpose_A<<<gridSize, blockSize>>>(d_hidden, d_output_grad, d_w2_grad, current_batch_size, OUTPUT_DIM, HIDDEN_DIM);
            
            // 计算隐藏层梯度：output_grad * W2^T
            gridSize = dim3((HIDDEN_DIM + blockSize.x - 1) / blockSize.x,
                           (current_batch_size + blockSize.y - 1) / blockSize.y);
            matmul_transpose_B<<<gridSize, blockSize>>>(d_output_grad, d_w2, d_hidden_grad, current_batch_size, OUTPUT_DIM, HIDDEN_DIM);
            
            // ReLU反向传播
            blocks = (current_batch_size * HIDDEN_DIM + threads - 1) / threads;
            compute_relu_backward<<<blocks, threads>>>(d_hidden_grad, d_hidden, current_batch_size * HIDDEN_DIM);
            
            // Dropout反向传播
            dropout_backward<<<blocks, threads>>>(d_hidden_grad, d_dropout_mask, current_batch_size * HIDDEN_DIM);
            
            // 计算W1梯度：X^T * hidden_grad
            gridSize = dim3((HIDDEN_DIM + blockSize.x - 1) / blockSize.x,
                           (INPUT_DIM + blockSize.y - 1) / blockSize.y);
            matmul_transpose_A<<<gridSize, blockSize>>>(d_X, d_hidden_grad, d_w1_grad, current_batch_size, HIDDEN_DIM, INPUT_DIM);
            
            // 更新参数（使用L2正则化）
            blocks = (INPUT_DIM * HIDDEN_DIM + threads - 1) / threads;
            sgd_update_l2<<<blocks, threads>>>(d_w1, d_w1_grad, LEARNING_RATE, L2_LAMBDA, INPUT_DIM * HIDDEN_DIM);
            
            // 计算偏置梯度（偏置不使用L2正则化）
            blocks = (HIDDEN_DIM + threads - 1) / threads;
            compute_bias_grad<<<blocks, threads>>>(d_hidden_grad, d_b1_grad, current_batch_size, HIDDEN_DIM);
            sgd_update<<<blocks, threads>>>(d_b1, d_b1_grad, LEARNING_RATE, HIDDEN_DIM);
            
            blocks = (HIDDEN_DIM * OUTPUT_DIM + threads - 1) / threads;
            sgd_update_l2<<<blocks, threads>>>(d_w2, d_w2_grad, LEARNING_RATE, L2_LAMBDA, HIDDEN_DIM * OUTPUT_DIM);
            
            blocks = (OUTPUT_DIM + threads - 1) / threads;
            compute_bias_grad<<<blocks, threads>>>(d_output_grad, d_b2_grad, current_batch_size, OUTPUT_DIM);
            sgd_update<<<blocks, threads>>>(d_b2, d_b2_grad, LEARNING_RATE, OUTPUT_DIM);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        total_training_time_ms += ms; // 累积训练时间
        
        total_loss /= num_batches;
        std::cout << "[Epoch " << epoch + 1 << "] Loss: " << total_loss << ", Time: " << ms << " ms" << std::endl;
    }

    std::cout << "\n=== Training Performance Summary ===" << std::endl;
    std::cout << "Total Training Time: " << total_training_time_ms / 1000.0 << " seconds" << std::endl;
    if (total_training_time_ms > 0) {
        double training_throughput = static_cast<double>(train_size * EPOCHS) / (total_training_time_ms / 1000.0);
        std::cout << "Training Throughput: " << training_throughput << " samples/sec" << std::endl;
    }


    // 推理部分，测试训练的MLP网络
    std::cout << "\n=== Testing Phase ===" << std::endl;
    std::vector<double> predictions;
    std::vector<double> actuals;
    double test_loss = 0.0;
    double mae = 0.0;  // 平均绝对误差
    
    // 测试所有测试样本以获得完整评估
    int test_samples_count = test_size;  // 重命名以避免与函数参数混淆
    
    auto inference_start_time = std::chrono::high_resolution_clock::now(); // 新增: 推理开始时间

    for (int i = train_size; i < train_size + test_samples_count; ++i) {
        std::vector<double> test_x(INPUT_DIM);
        for (int j = 0; j < INPUT_DIM; ++j) {
            test_x[j] = X[i * INPUT_DIM + j];
        }
        
        hipMemcpy(d_X, test_x.data(), INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
        
        // 前向传播
        dim3 blockSize(16, 16);
        dim3 gridSize((HIDDEN_DIM + blockSize.x - 1) / blockSize.x, 1);
        matmul<<<gridSize, blockSize>>>(d_X, d_w1, d_hidden, 1, HIDDEN_DIM, INPUT_DIM);
        
        int threads = 256;
        int blocks = (HIDDEN_DIM + threads - 1) / threads;
        add_bias<<<blocks, threads>>>(d_hidden, d_b1, 1, HIDDEN_DIM);
        relu_activation<<<blocks, threads>>>(d_hidden, HIDDEN_DIM);
        // 推理模式：不使用dropout
        
        gridSize = dim3((OUTPUT_DIM + blockSize.x - 1) / blockSize.x, 1);
        matmul<<<gridSize, blockSize>>>(d_hidden, d_w2, d_output, 1, OUTPUT_DIM, HIDDEN_DIM);
        
        blocks = (OUTPUT_DIM + threads - 1) / threads;
        add_bias<<<blocks, threads>>>(d_output, d_b2, 1, OUTPUT_DIM);
        
        double prediction;
        hipMemcpy(&prediction, d_output, sizeof(double), hipMemcpyDeviceToHost);
        predictions.push_back(prediction);
        
        double actual = y[i];
        actuals.push_back(actual);
        
        double diff = prediction - actual;
        test_loss += diff * diff;
        mae += std::abs(diff);
        
        // 只显示前10个样本的详细结果，避免输出过多
        if (i < train_size + 10) {
            std::cout << "Sample " << (i - train_size + 1) << " - Normalized Predicted: " << prediction 
                      << ", Normalized Actual: " << actual << ", Error: " << diff << std::endl;
        }
    }
    
    auto inference_end_time = std::chrono::high_resolution_clock::now(); // 新增: 推理结束时间
    double total_inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end_time - inference_start_time).count();

    // 计算评估指标
    test_loss /= test_samples_count;  // MSE
    mae /= test_samples_count;        // MAE
    
    // 计算决定系数 R²
    double mean_actual = 0.0;
    for (double val : actuals) mean_actual += val;
    mean_actual /= actuals.size();
    
    double ss_tot = 0.0, ss_res = 0.0;
    for (int i = 0; i < actuals.size(); ++i) {
        ss_tot += (actuals[i] - mean_actual) * (actuals[i] - mean_actual);
        ss_res += (actuals[i] - predictions[i]) * (actuals[i] - predictions[i]);
    }
    double r2 = 1.0 - (ss_res / ss_tot);
    
    std::cout << "\n=== Test Results Summary ===" << std::endl;
    std::cout << "Total test samples: " << test_samples_count << " (100% of test set)" << std::endl;
    std::cout << "Mean Squared Error (MSE): " << test_loss << std::endl;
    std::cout << "Root Mean Squared Error (RMSE): " << std::sqrt(test_loss) << std::endl;
    std::cout << "Mean Absolute Error (MAE): " << mae << std::endl;
    std::cout << "R² Score: " << r2 << std::endl;

    std::cout << "\n=== Inference Performance Summary ===" << std::endl;
    std::cout << "Total Inference Time: " << total_inference_time_ms / 1000.0 << " seconds for " << test_samples_count << " samples" << std::endl;
    if (test_samples_count > 0 && total_inference_time_ms > 0) {
        double avg_inference_latency_ms = total_inference_time_ms / test_samples_count;
        std::cout << "Average Inference Latency: " << avg_inference_latency_ms << " ms/sample" << std::endl;
        double inference_throughput = static_cast<double>(test_samples_count) / (total_inference_time_ms / 1000.0);
        std::cout << "Inference Throughput: " << inference_throughput << " samples/sec" << std::endl;
    }
    
    // 反归一化显示真实值（如果需要）
    std::vector<double> real_predictions = predictions;
    std::vector<double> real_actuals = actuals;
    denormalize_data(real_predictions, min_val, max_val);
    denormalize_data(real_actuals, min_val, max_val);
    
    std::cout << "\n=== Real Values (Denormalized) Sample (First 10) ===" << std::endl;
    for (int i = 0; i < std::min(10, (int)real_predictions.size()); ++i) {
        std::cout << "Real Predicted: " << real_predictions[i] 
                  << ", Real Actual: " << real_actuals[i] 
                  << ", Real Error: " << (real_predictions[i] - real_actuals[i]) << std::endl;
    }
    
    // 保存训练的神经网络
    hipMemcpy(w1.data(), d_w1, w1.size() * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(b1.data(), d_b1, b1.size() * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(w2.data(), d_w2, w2.size() * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(b2.data(), d_b2, b2.size() * sizeof(double), hipMemcpyDeviceToHost);
    
    save_model(w1, b1, w2, b2, "mlp_model.txt");
    
    // 新增: 保存预测结果到CSV文件
    save_predictions_to_csv(actuals, predictions, real_actuals, real_predictions, "predictions_vs_actuals.csv");

    // 清理GPU内存
    hipFree(d_X); hipFree(d_y); hipFree(d_w1); hipFree(d_b1);
    hipFree(d_w2); hipFree(d_b2); hipFree(d_hidden); hipFree(d_output);
    hipFree(d_output_grad); hipFree(d_hidden_grad); hipFree(d_w1_grad);
    hipFree(d_b1_grad); hipFree(d_w2_grad); hipFree(d_b2_grad); hipFree(d_loss);
    hipFree(d_dropout_mask); hipFree(d_l2_loss);
    delete[] h_loss; delete[] h_l2_loss;
    
    std::cout << "Training completed successfully!" << std::endl;
    return 0;
}
