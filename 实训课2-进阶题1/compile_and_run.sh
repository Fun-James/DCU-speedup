#!/bin/bash

# 编译脚本
echo "正在编译 MLP 前向传播程序..."
hipcc sourcefile_mlp_forward.cpp -o mlp_forward

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo "运行程序..."
    echo "================================"
    ./mlp_forward
    echo "================================"
    echo "程序执行完成。"
else
    echo "编译失败！"
    exit 1
fi
