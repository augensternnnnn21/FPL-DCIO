#!/bin/zsh

# 添加路径验证
DATA_PATH="$HOME/Desktop/FPL-DCIO/data"
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ 数据目录不存在: $DATA_PATH"
    exit 1
fi

# 增强Python路径设置
export PYTHONPATH="$PYTHONPATH:$HOME/Desktop/FPL-DCIO/src"
cd "$HOME/Desktop/FPL-DCIO/src" || exit 1

# 添加执行前清理
find . -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -exec rm -f {} +

# 运行主程序
python3 run_experiment.py --data_path "$DATA_PATH"