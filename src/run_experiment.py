import os
import torch
from pathlib import Path
from config import ExperimentConfig, get_default_config, ModelConfig
from experiments import ExperimentRunner
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='FPL-DCIO 实验运行器')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, required=True,
                      help='数据目录路径')
    parser.add_argument('--output_path', type=str, required=True,
                      help='输出目录路径')
    parser.add_argument('--train_seasons', nargs='+', type=str, required=True,
                      help='训练赛季列表')
    parser.add_argument('--val_season', type=str, required=True,
                      help='验证赛季')
    parser.add_argument('--test_season', type=str, required=True,
                      help='测试赛季')
    
    # 环境相关参数
    parser.add_argument('--budget', type=float, default=100.0,
                      help='预算限制')
    parser.add_argument('--max_players', type=int, default=15,
                      help='最大球员数量')
    
    # 训练相关参数
    parser.add_argument('--num_episodes', type=int, default=1000,
                      help='训练回合数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='折扣因子')
    
    # 模型相关参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                      help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                      help='层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout率')
    parser.add_argument('--bidirectional', type=str, default='true',
                      help='是否使用双向LSTM')
    parser.add_argument('--use_attention', type=str, default='true',
                      help='是否使用注意力机制')
    
    args = parser.parse_args()
    
    # 处理布尔值参数
    args.bidirectional = args.bidirectional.lower() == 'true'
    args.use_attention = args.use_attention.lower() == 'true'
    
    # 创建模型配置
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        use_attention=args.use_attention
    )
    
    # 创建实验配置
    config = ExperimentConfig(
        data_path=args.data_path,
        train_seasons=args.train_seasons,
        val_season=args.val_season,
        test_season=args.test_season,
        budget=args.budget,
        max_players=args.max_players,
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        output_path=args.output_path,
        model=model_config
    )
    
    return config

def main():
    print("\nDEBUG: Starting main...")
    
    # 使用正确的配置初始化方式
    config = parse_args()
    
    # 创建实验运行器
    try:
        runner = ExperimentRunner(config)
        runner.run_experiments()
    except Exception as e:
        print(f"Error running experiments: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()