#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 测试模型
python3 src/run_experiment.py \
    --mode test \
    --model_path experiments/best_model.pth \
    --test_season "2020-21" \
    --num_episodes 10 \
    --use_ilp True \
    --use_synergy True 