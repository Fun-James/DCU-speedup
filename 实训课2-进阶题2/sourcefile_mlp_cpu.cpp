#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>

// 编译文件
// g++ -O3 -std=c++11 sourcefile_mlp_cpu.cpp -o mlp_full_cpu
// 执行文件
// ./mlp_full_cpu

// 预定义参数，可根据需求修改 - 基于时序分析优化
#define INPUT_DIM 50       // 基于自相关分析的最佳窗口大小
#define HIDDEN_DIM 96      // 适当增加隐藏层以处理更多输入特征
#define OUTPUT_DIM 1
#define BATCH_SIZE 128     // 减小批次大小以适应更大的输入维度
#define EPOCHS 2000        // 增加训练轮次以充分学习时序模式
#define LEARNING_RATE 3e-4 // 调整学习率以适应新的网络规模
#define DROPOUT_RATE 0.12  // 轻微降低dropout，保留更多时序信息
#define L2_LAMBDA 2e-5     // 适当调整L2正则化     

// CPU实现的矩阵乘法 C = A * B
void matmul(const double* A, const double* B, double* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// 计算输出层梯度
void compute_output_grad(const double* pred, const double* target, double* grad, int size) {
    for (int idx = 0; idx < size; ++idx) {
        grad[idx] = 2.0 * (pred[idx] - target[idx]) / size;  // MSE gradient
    }
}

// ReLU反向传播
void compute_relu_backward(double* delta, const double* activ, int size) {
    for (int idx = 0; idx < size; ++idx) {
        delta[idx] = (activ[idx] > 0.0) ? delta[idx] : 0.0;
    }
}

// 计算MSE损失
void compute_mse_loss(const double* pred, const double* target, double* loss, int size) {
    for (int idx = 0; idx < size; ++idx) {
        double diff = pred[idx] - target[idx];
        loss[idx] = diff * diff;
    }
}

// SGD更新
void sgd_update(double* weights, const double* grad, double lr, int size) {
    for (int idx = 0; idx < size; ++idx) {
        weights[idx] -= lr * grad[idx];
    }
}

// L2正则化的SGD更新（包含权重衰减）
void sgd_update_l2(double* weights, const double* grad, double lr, double l2_lambda, int size) {
    for (int idx = 0; idx < size; ++idx) {
        // 梯度 = 原梯度 + L2正则化项
        // 权重更新 = 权重 - 学习率 * (梯度 + λ * 权重)
        weights[idx] -= lr * (grad[idx] + l2_lambda * weights[idx]);
    }
}

// 计算L2正则化损失
void compute_l2_loss(const double* weights, double* l2_loss, int size) {
    for (int idx = 0; idx < size; ++idx) {
        l2_loss[idx] = weights[idx] * weights[idx];
    }
}

// 转置矩阵乘法：A^T * B
void matmul_transpose_A(const double* A, const double* B, double* C, int M, int N, int K) {
    for (int row = 0; row < K; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int m = 0; m < M; ++m) {
                sum += A[m * K + row] * B[m * N + col];  // A转置
            }
            C[row * N + col] = sum;
        }
    }
}

// 矩阵乘法：A * B^T
void matmul_transpose_B(const double* A, const double* B, double* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < K; ++col) {
            double sum = 0.0;
            for (int n = 0; n < N; ++n) {
                sum += A[row * N + n] * B[col * N + n];  // B转置
            }
            C[row * K + col] = sum;
        }
    }
}

// 计算偏置梯度（对批次求和）
void compute_bias_grad(const double* grad, double* bias_grad, int batch_size, int feature_size) {
    for (int idx = 0; idx < feature_size; ++idx) {
        double sum = 0.0;
        for (int b = 0; b < batch_size; ++b) {
            sum += grad[b * feature_size + idx];
        }
        bias_grad[idx] = sum;
    }
}

// 简单的随机数生成器类
class SimpleRandom {
private:
    unsigned int seed;
public:
    SimpleRandom(unsigned int s) : seed(s) {}
    
    double next() {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return (double)seed / 0x7fffffff;
    }
};

// Dropout前向传播 (训练模式)
void dropout_forward_train(double* data, double* mask, 
                          double dropout_rate, int size, unsigned int epoch_seed) {
    SimpleRandom rng(epoch_seed);
    for (int idx = 0; idx < size; ++idx) {
        double rand_val = rng.next();
        
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
void dropout_backward(double* grad, const double* mask, int size) {
    for (int idx = 0; idx < size; ++idx) {
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
}

// 数据归一化处理
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
}

// 数据反归一化处理
void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
}

// ReLU激活函数
void relu_activation(double* data, int size) {
    for (int idx = 0; idx < size; ++idx) {
        data[idx] = std::max(0.0, data[idx]);
    }
}

// 添加偏置
void add_bias(double* data, const double* bias, int batch_size, int feature_size) {
    for (int idx = 0; idx < batch_size * feature_size; ++idx) {
        int feature_idx = idx % feature_size;
        data[idx] += bias[feature_idx];
    }
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
    
    // 分配CPU内存
    std::vector<double> hidden(BATCH_SIZE * HIDDEN_DIM);
    std::vector<double> output(BATCH_SIZE * OUTPUT_DIM);
    std::vector<double> output_grad(BATCH_SIZE * OUTPUT_DIM);
    std::vector<double> hidden_grad(BATCH_SIZE * HIDDEN_DIM);
    std::vector<double> w1_grad(INPUT_DIM * HIDDEN_DIM);
    std::vector<double> b1_grad(HIDDEN_DIM);
    std::vector<double> w2_grad(HIDDEN_DIM * OUTPUT_DIM);
    std::vector<double> b2_grad(OUTPUT_DIM);
    std::vector<double> loss(BATCH_SIZE);
    std::vector<double> dropout_mask(BATCH_SIZE * HIDDEN_DIM);
    std::vector<double> l2_loss(std::max(INPUT_DIM * HIDDEN_DIM, HIDDEN_DIM * OUTPUT_DIM));
    
    std::cout << "Starting CPU MLP training..." << std::endl;
    
    long long total_training_duration_ms = 0; // 用于累积总训练时间

    // 训练MLP网络
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
            
            // 前向传播
            // 隐藏层：X * W1 + b1
            matmul(batch_X.data(), w1.data(), hidden.data(), current_batch_size, HIDDEN_DIM, INPUT_DIM);
            
            // 添加偏置
            add_bias(hidden.data(), b1.data(), current_batch_size, HIDDEN_DIM);
            
            // ReLU激活
            relu_activation(hidden.data(), current_batch_size * HIDDEN_DIM);
            
            // 应用Dropout (训练模式)
            dropout_forward_train(hidden.data(), dropout_mask.data(), 
                                 DROPOUT_RATE, current_batch_size * HIDDEN_DIM, epoch * 1000 + start_idx);
            
            // 输出层：hidden * W2 + b2
            matmul(hidden.data(), w2.data(), output.data(), current_batch_size, OUTPUT_DIM, HIDDEN_DIM);
            
            add_bias(output.data(), b2.data(), current_batch_size, OUTPUT_DIM);
            
            // 计算损失
            compute_mse_loss(output.data(), batch_y.data(), loss.data(), current_batch_size);
            double mse_loss = compute_total_loss(loss.data(), current_batch_size);
            
            // 计算L2正则化损失
            double l2_reg_loss = 0.0;
            // W1的L2损失
            compute_l2_loss(w1.data(), l2_loss.data(), INPUT_DIM * HIDDEN_DIM);
            for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; ++i) {
                l2_reg_loss += l2_loss[i];
            }
            
            // W2的L2损失
            compute_l2_loss(w2.data(), l2_loss.data(), HIDDEN_DIM * OUTPUT_DIM);
            for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; ++i) {
                l2_reg_loss += l2_loss[i];
            }
            
            // 总损失 = MSE损失 + λ * L2损失
            total_loss += mse_loss + L2_LAMBDA * l2_reg_loss;
            
            // 反向传播
            // 输出层梯度
            compute_output_grad(output.data(), batch_y.data(), output_grad.data(), current_batch_size);
            
            // 计算W2梯度：hidden^T * output_grad
            matmul_transpose_A(hidden.data(), output_grad.data(), w2_grad.data(), 
                              current_batch_size, OUTPUT_DIM, HIDDEN_DIM);
            
            // 计算隐藏层梯度：output_grad * W2^T
            matmul_transpose_B(output_grad.data(), w2.data(), hidden_grad.data(), 
                              current_batch_size, OUTPUT_DIM, HIDDEN_DIM);
            
            // ReLU反向传播
            compute_relu_backward(hidden_grad.data(), hidden.data(), current_batch_size * HIDDEN_DIM);
            
            // Dropout反向传播
            dropout_backward(hidden_grad.data(), dropout_mask.data(), current_batch_size * HIDDEN_DIM);
            
            // 计算W1梯度：X^T * hidden_grad
            matmul_transpose_A(batch_X.data(), hidden_grad.data(), w1_grad.data(), 
                              current_batch_size, HIDDEN_DIM, INPUT_DIM);
            
            // 更新参数（使用L2正则化）
            sgd_update_l2(w1.data(), w1_grad.data(), LEARNING_RATE, L2_LAMBDA, INPUT_DIM * HIDDEN_DIM);
            
            // 计算偏置梯度（偏置不使用L2正则化）
            compute_bias_grad(hidden_grad.data(), b1_grad.data(), current_batch_size, HIDDEN_DIM);
            sgd_update(b1.data(), b1_grad.data(), LEARNING_RATE, HIDDEN_DIM);
            
            sgd_update_l2(w2.data(), w2_grad.data(), LEARNING_RATE, L2_LAMBDA, HIDDEN_DIM * OUTPUT_DIM);
            
            compute_bias_grad(output_grad.data(), b2_grad.data(), current_batch_size, OUTPUT_DIM);
            sgd_update(b2.data(), b2_grad.data(), LEARNING_RATE, OUTPUT_DIM);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        total_training_duration_ms += ms; // 累积训练时间
        
        total_loss /= num_batches;
        std::cout << "[Epoch " << epoch + 1 << "] Loss: " << total_loss << ", Time: " << ms << " ms" << std::endl;
    }

    std::cout << "\n=== Training Summary ===" << std::endl;
    double total_training_time_seconds = total_training_duration_ms / 1000.0;
    std::cout << "Total Training Time: " << total_training_time_seconds << " seconds" << std::endl;
    if (total_training_time_seconds > 0) {
        double training_throughput = (double)train_size * EPOCHS / total_training_time_seconds;
        std::cout << "Training Throughput: " << training_throughput << " samples/sec" << std::endl;
    } else {
        std::cout << "Training Throughput: N/A (training time is zero)" << std::endl;
    }

    // 推理部分，测试训练的MLP网络
    std::cout << "\n=== Testing Phase ===" << std::endl;
    std::vector<double> predictions;
    std::vector<double> actuals;
    double test_loss = 0.0;
    double mae = 0.0;  // 平均绝对误差
    
    // 测试所有测试样本以获得完整评估
    int test_samples = test_size;  // 测试全部测试集
    
    auto inference_start_time = std::chrono::high_resolution_clock::now(); // 推理开始时间

    for (int i = train_size; i < train_size + test_samples; ++i) {
        std::vector<double> test_x(INPUT_DIM);
        for (int j = 0; j < INPUT_DIM; ++j) {
            test_x[j] = X[i * INPUT_DIM + j];
        }
        
        // 前向传播
        std::vector<double> test_hidden(HIDDEN_DIM);
        std::vector<double> test_output(OUTPUT_DIM);
        
        matmul(test_x.data(), w1.data(), test_hidden.data(), 1, HIDDEN_DIM, INPUT_DIM);
        
        add_bias(test_hidden.data(), b1.data(), 1, HIDDEN_DIM);
        relu_activation(test_hidden.data(), HIDDEN_DIM);
        // 推理模式：不使用dropout
        
        matmul(test_hidden.data(), w2.data(), test_output.data(), 1, OUTPUT_DIM, HIDDEN_DIM);
        
        add_bias(test_output.data(), b2.data(), 1, OUTPUT_DIM);
        
        double prediction = test_output[0];
        predictions.push_back(prediction);
        
        double actual = y[i];
        actuals.push_back(actual);
        
        double diff = prediction - actual;
        test_loss += diff * diff;
        mae += std::abs(diff);
        
        // 只显示前10个样本的详细结果，避免输出过多
        if (i < train_size + 10) {
            std::cout << "Sample " << (i - train_size + 1) << " - Predicted: " << prediction 
                      << ", Actual: " << actual << ", Error: " << diff << std::endl;
        }
    }
    
    auto inference_end_time = std::chrono::high_resolution_clock::now(); // 推理结束时间
    auto inference_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end_time - inference_start_time).count();

    // 计算评估指标
    test_loss /= test_samples;  // MSE
    mae /= test_samples;        // MAE
    
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
    std::cout << "Total test samples: " << test_samples << " (100% of test set)" << std::endl;
    
    double total_inference_time_seconds = inference_duration_ms / 1000.0;
    std::cout << "Total Inference Time: " << total_inference_time_seconds << " seconds" << std::endl;
    if (test_samples > 0) {
        double inference_latency_ms_per_sample = (double)inference_duration_ms / test_samples;
        std::cout << "Inference Latency: " << inference_latency_ms_per_sample << " ms/sample" << std::endl;
    } else {
        std::cout << "Inference Latency: N/A (no test samples)" << std::endl;
    }
    if (total_inference_time_seconds > 0) {
        double inference_throughput_sps = (double)test_samples / total_inference_time_seconds;
        std::cout << "Inference Throughput: " << inference_throughput_sps << " samples/sec" << std::endl;
    } else {
        std::cout << "Inference Throughput: N/A (inference time is zero)" << std::endl;
    }
    
    std::cout << "Mean Squared Error (MSE): " << test_loss << std::endl;
    std::cout << "Root Mean Squared Error (RMSE): " << std::sqrt(test_loss) << std::endl;
    std::cout << "Mean Absolute Error (MAE): " << mae << std::endl;
    std::cout << "R² Score: " << r2 << std::endl;
    
    // 反归一化显示真实值（如果需要）
    std::vector<double> real_predictions = predictions;
    std::vector<double> real_actuals = actuals;
    denormalize_data(real_predictions, min_val, max_val);
    denormalize_data(real_actuals, min_val, max_val);
    
    std::cout << "\n=== Real Values (Denormalized) Sample ===" << std::endl;
    for (int i = 0; i < std::min(10, (int)real_predictions.size()); ++i) {
        std::cout << "Real Predicted: " << real_predictions[i] 
                  << ", Real Actual: " << real_actuals[i] 
                  << ", Real Error: " << (real_predictions[i] - real_actuals[i]) << std::endl;
    }
    
    // 保存训练的神经网络
    save_model(w1, b1, w2, b2, "mlp_model_cpu.txt");
    
    std::cout << "CPU Training completed successfully!" << std::endl;
    return 0;
}
