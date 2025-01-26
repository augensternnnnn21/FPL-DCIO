#!/bin/bash

# 添加缓存清理命令
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH: $PYTHONPATH"
echo "Current directory: $(pwd)"

# 运行实验
echo "Running experiments..."
PYTHONPATH=.:$PYTHONPATH python3 run_experiment.py