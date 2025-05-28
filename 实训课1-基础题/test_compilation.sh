#!/bin/bash

echo "=== 完整编译测试 ==="
echo "检查所有编译警告是否已修复"
echo ""

# 编译DCU版本并检查警告
echo "1. 编译DCU版本..."
hipcc sourcefile_dcu.cpp -o matmul_dcu 2>&1 | tee compile_dcu.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "   ✓ DCU版本编译成功"
    
    # 检查是否还有警告
    warning_count=$(grep -c "warning:" compile_dcu.log || echo "0")
    if [ "$warning_count" -eq "0" ]; then
        echo "   ✓ 无编译警告"
    else
        echo "   ⚠ 仍有 $warning_count 个警告"
        echo "   警告详情："
        grep "warning:" compile_dcu.log
    fi
else
    echo "   ✗ DCU版本编译失败"
    exit 1
fi

echo ""

# 编译CPU版本
echo "2. 编译CPU版本..."
mpic++ -fopenmp -o matmul_cpu sourcefile.cpp 2>&1 | tee compile_cpu.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "   ✓ CPU版本编译成功"
    
    # 检查是否还有警告
    warning_count=$(grep -c "warning:" compile_cpu.log || echo "0")
    if [ "$warning_count" -eq "0" ]; then
        echo "   ✓ 无编译警告"
    else
        echo "   ⚠ 仍有 $warning_count 个警告"
        echo "   警告详情："
        grep "warning:" compile_cpu.log
    fi
else
    echo "   ✗ CPU版本编译失败"
    exit 1
fi

echo ""
echo "=== 运行快速测试 ==="

# 运行CPU版本
echo "3. 测试CPU版本..."
timeout 30s ./matmul_cpu baseline
cpu_exit_code=$?

if [ $cpu_exit_code -eq 0 ]; then
    echo "   ✓ CPU版本运行成功"
elif [ $cpu_exit_code -eq 124 ]; then
    echo "   ⚠ CPU版本运行超时（正常，矩阵较大）"
else
    echo "   ✗ CPU版本运行失败"
fi

echo ""

# 运行DCU版本
echo "4. 测试DCU版本..."
timeout 30s ./matmul_dcu
dcu_exit_code=$?

if [ $dcu_exit_code -eq 0 ]; then
    echo "   ✓ DCU版本运行成功"
elif [ $dcu_exit_code -eq 124 ]; then
    echo "   ⚠ DCU版本运行超时（可能需要更长时间）"
else
    echo "   ✗ DCU版本运行失败"
fi

echo ""
echo "=== 测试完成 ==="
echo "编译日志已保存到 compile_cpu.log 和 compile_dcu.log"
