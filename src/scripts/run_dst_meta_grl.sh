#!/bin/bash

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 打印环境变量
echo "PYTHONPATH: $PYTHONPATH"
echo "Current directory: $(pwd)"

# 预处理数据
echo "Preprocessing data..."
python3 preprocess.py

# 运行实验
echo "Running experiments..."
python3 run_experiment.py

# 结果将保存在 experiments/ 目录下 