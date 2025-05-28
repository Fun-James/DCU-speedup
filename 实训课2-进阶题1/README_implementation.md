# MLP 前向传播实现说明

## 🎯 项目概述
这是一个基于 HIP 的多层感知机（MLP）前向传播实现，在 AMD DCU 加速卡上运行。

## 📁 文件结构
- `sourcefile_mlp_forward.cpp` - 主要实现文件
- `compile_and_run.ps1` - PowerShell 编译脚本
- `README_implementation.md` - 本说明文档

## 🏗️ 网络架构
- **输入层**: 1024 × 10 (批次大小 × 输入维度)
- **隐藏层**: 10 × 20 权重矩阵 + 偏置，ReLU 激活
- **输出层**: 20 × 5 权重矩阵 + 偏置，无激活

## 🔧 核心功能实现

### 1. 矩阵乘法内核 (`matmul_kernel`)
```cpp
__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int N, int K)
```
- 实现 GPU 并行矩阵乘法：C = A × B
- 使用 2D 线程块布局 (16×16)
- 每个线程计算结果矩阵的一个元素

### 2. ReLU 激活函数 (`relu_kernel`)
```cpp
__global__ void relu_kernel(double* A, int size)
```
- 对输入进行 ReLU 激活：max(0, x)
- 1D 线程布局，每个线程处理一个元素

### 3. 偏置加法内核 (`add_bias_kernel`)
```cpp
__global__ void add_bias_kernel(double* A, const double* bias, int M, int N)
```
- 将偏置向量广播加到矩阵的每一行
- 支持批量处理

## 🚀 编译和运行

### 方法1：使用 PowerShell 脚本
```powershell
.\compile_and_run.ps1
```

### 方法2：手动编译
```powershell
hipcc sourcefile_mlp_forward.cpp -o mlp_forward.exe
.\mlp_forward.exe
```

### 方法3：性能分析
```powershell
hipprof .\mlp_forward.exe
```

## 📊 程序输出
程序会输出：
1. 网络配置信息
2. 各阶段执行状态
3. GPU 计算时间
4. 前 5 个样本的输出结果
5. 总执行时间

## 🔍 性能特性

### 内存管理
- 使用 `HIP_CHECK` 宏进行错误检查
- 统一的内存分配和释放
- 高效的主机-设备数据传输

### 计算优化
- 2D 线程块布局优化矩阵乘法
- 合理的线程块大小 (16×16 和 256)
- 内核执行同步确保计算正确性

### 监控和调试
- 详细的执行状态输出
- 微秒级的性能计时
- 错误检查和异常处理

## 📈 计算流程
1. **数据初始化**: 随机生成输入和权重
2. **内存分配**: 在 GPU 上分配内存空间
3. **数据传输**: 将数据从 CPU 复制到 GPU
4. **隐藏层计算**: 
   - 矩阵乘法：X × W1
   - 加偏置：+ B1
   - ReLU 激活
5. **输出层计算**:
   - 矩阵乘法：H × W2
   - 加偏置：+ B2
6. **结果传输**: 将结果从 GPU 复制回 CPU
7. **内存清理**: 释放 GPU 内存

## 🎯 作业要求完成情况
- ✅ 实现基于矩阵乘法的 MLP 前向传播
- ✅ 支持批处理 (batch_size = 1024)
- ✅ 使用 HIP 在 DCU 上运行
- ✅ 双精度浮点数计算
- ✅ 矩阵乘法优化
- ✅ 性能评测功能
- ✅ 错误检查和调试信息

## 🔧 扩展建议
1. **进一步优化**:
   - 使用共享内存优化矩阵乘法
   - 实现 tiled 矩阵乘法算法
   - 使用 hipBLAS 库加速

2. **功能扩展**:
   - 添加反向传播
   - 支持多种激活函数
   - 实现批量归一化

3. **性能分析**:
   - 使用 rocprof 进行详细性能分析
   - 内存带宽利用率分析
   - 不同矩阵大小的性能对比
