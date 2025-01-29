#!/bin/bash

# 设置项目根目录
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
echo "DEBUG: PROJECT_ROOT: $PROJECT_ROOT"

# 激活Python环境（如果需要）
# source $PROJECT_ROOT/venv/bin/activate

# 设置PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 创建输出目录
mkdir -p "$PROJECT_ROOT/experiments"

# 运行实验
python3 $PROJECT_ROOT/src/run_experiment.py \
    --data_path "$PROJECT_ROOT/data" \
    --output_path "$PROJECT_ROOT/experiments" \
    --budget 100.0 \
    --max_players 15 \
    --num_episodes 1000 \
    --learning_rate 0.001 \
    --gamma 0.99 \
    --train_seasons "2016-17" "2017-18" "2018-19" \
    --val_season "2019-20" \
    --test_season "2020-21" \
    --hidden_dim 128 \
    --num_layers 3 \
    --dropout 0.1 \
    --bidirectional true \
    --use_attention true \
    --use_ilp True \
    --use_synergy True \
    --use_triple_captain=True \
    --use_bench_boost=False