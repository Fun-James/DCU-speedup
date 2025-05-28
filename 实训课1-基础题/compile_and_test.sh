#!/bin/bash

echo "=== 编译和测试矩阵乘法实现 ==="

# 编译普通C++版本
echo "1. 编译普通C++版本 (CPU)..."
mpic++ -fopenmp -O3 -o matmul_cpu sourcefile.cpp
if [ $? -eq 0 ]; then
    echo "   ✓ CPU版本编译成功"
else
    echo "   ✗ CPU版本编译失败"
    exit 1
fi


# 编译DCU版本
echo ""
echo "2. 编译DCU版本..."
hipcc sourcefile_dcu.cpp -o matmul_dcu
if [ $? -eq 0 ]; then
    echo "   ✓ DCU版本编译成功"
else
    echo "   ✗ DCU版本编译失败"
    exit 1
fi

echo ""
echo "=== 运行测试 ==="

# 测试普通C++版本
echo ""
echo "3. 测试 CPU baseline 版本..."
./matmul_cpu baseline

echo ""
echo "4. 测试 CPU OpenMP 版本..."
./matmul_cpu openmp

echo ""
echo "5. 测试 CPU 子块并行 (Block Tiling) 版本..."
./matmul_cpu block



echo ""
echo "6. 测试 CPU MPI 版本 (使用 8 个进程)..."
mpirun --allow-run-as-root -np 8 ./matmul_cpu mpi

echo ""
echo "7. 测试 CPU MPI+OpenMP子块并行版本..."
mpirun --allow-run-as-root -np 8 ./matmul_cpu other


echo ""
echo "8. 测试 DCU 版本..."
./matmul_dcu

echo ""
echo "=== 测试完成 ==="
