#!/bin/bash
# 预处理所有数据
echo "预处理数据..."
python3 src/old/preprocess.py

# 按时间序列训练模型
echo "开始训练模型..."
python3 src/old/train.py

# 运行基准测试
echo "运行基准测试..."
python3 src/old/benchmark.py

# 运行消融实验
echo "运行消融实验..."
python3 src/old/ablation.py

