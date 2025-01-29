# FPL-DCIO: Fantasy Premier League Decision and Optimization

## Project Overview
A deep reinforcement learning framework for FPL team selection, combining Graph Neural Networks with Integer Linear Programming.

## Architecture
- Graph Attention Networks (GAT) for modeling player synergies
- Integer Linear Programming (ILP) for constraint satisfaction
- Reinforcement Learning for team selection optimization
- Cross-season generalization evaluation

## Key Features
- Automated team selection considering:
  - Budget constraints (≤100M)
  - Team composition (2-5-5-3)
  - Maximum 3 players per club
- Special cards support:
  - Triple Captain
  - Bench Boost
- Player synergy modeling through GAT
- Robust constraint handling via ILP

## Project Structure
src/
├── data/
│   ├── data_loader.py      # 数据加载
│   └── preprocessor.py     # 数据预处理
├── models/
│   ├── gat_fusion.py       # 图注意力融合
│   ├── neural_ilp.py       # ILP求解器
│   ├── st_conv.py          # 主模型
│   └── generalization.py   # 泛化性评估
├── utils/
│   ├── logger.py           # 日志工具
│   ├── statistical_analysis.py  # 统计分析
│   └── visualization.py    # 可视化工具
├── scripts/
│   └── run_dst_meta_grl.sh # 运行脚本
├── environment.py          # 环境实现
├── experiments.py          # 实验运行器
└── run_experiment.py       # 主运行文件
