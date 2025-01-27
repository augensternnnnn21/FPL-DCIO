#!/bin/bash
# 预处理数据
python3 src/old/preprocess.py
# 快速训练（50个Episode）
python3 src/old/train.py 
