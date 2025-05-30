#!/bin/bash

# MLP 神经网络 - 简化编译运行脚本
echo "=========================================="
echo "MLP GPU 神经网络 - 编译运行"
echo "=========================================="

# 项目配置
SOURCE_FILE="sourcefile_mlp.cpp"
OUTPUT_BINARY="mlp_full_dcu"
DATA_FILE="starlink_bw.json"

# 检查必要文件
echo "检查必要文件..."
if [ ! -f "$SOURCE_FILE" ]; then
    echo "错误: 源文件不存在: $SOURCE_FILE"
    exit 1
fi
if [ ! -f "$DATA_FILE" ]; then # 确保检查数据文件，因为C++程序需要它
    echo "错误: 数据文件不存在: $DATA_FILE"
    exit 1
fi
echo "✓ 所有必要文件存在"

# 检查HIP编译器
if ! command -v hipcc &> /dev/null; then
    echo "错误: hipcc编译器未找到!"
    echo "请确保HIP SDK已正确安装"
    exit 1
fi
echo "✓ HIP编译器已找到"

# 清理旧文件
if [ -f "$OUTPUT_BINARY" ]; then
    rm "$OUTPUT_BINARY"
fi

# 编译项目（减少警告输出）
echo "编译中..."
COMPILE_FLAGS="-O3 -std=c++11" # 添加 -Wno-return-type

if hipcc $COMPILE_FLAGS "$SOURCE_FILE" -o "$OUTPUT_BINARY"; then
    echo "✓ 编译成功!"
else
    echo "错误: 编译失败。请检查上面的错误信息。"
    exit 1
fi

# 运行程序
echo "=========================================="
echo "开始训练..."
echo "=========================================="
./"$OUTPUT_BINARY"

echo "=================================================="
echo "程序执行完毕。"
echo "=================================================="
