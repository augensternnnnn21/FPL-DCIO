#!/bin/bash

# 设置项目根目录
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"

# 设置Python路径
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 创建必要的目录
mkdir -p $PROJECT_ROOT/data/processed
mkdir -p $PROJECT_ROOT/experiments/logs
mkdir -p $PROJECT_ROOT/models/saved

# 步骤1: 数据预处理
echo "Step 1: Preprocessing data..."
python3 $PROJECT_ROOT/src/old/preprocess.py

# 步骤2: 运行实验
echo "Step 2: Running experiments..."
python3 $PROJECT_ROOT/src/run_experiment.py 