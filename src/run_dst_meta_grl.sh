#!/bin/bash

# 设置环境变量
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 预处理数据
echo "Preprocessing data..."
python3 preprocess.py

# 运行实验
echo "Running experiments..."
python3 run_experiment.py

# 结果将保存在 experiments/ 目录下 