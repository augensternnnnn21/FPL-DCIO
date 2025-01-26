import argparse
from experiments import ExperimentRunner
from config import ExperimentConfig, ModelConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--budget', type=float, default=100.0)
    parser.add_argument('--max_players', type=int, default=15)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--train_seasons', nargs='+', required=True)
    parser.add_argument('--val_season', type=str, required=True)
    parser.add_argument('--test_season', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--use_attention', type=bool, default=True)
    parser.add_argument('--use_ilp', type=bool, default=True)
    parser.add_argument('--use_synergy', type=bool, default=True)
    parser.add_argument('--use_triple_captain', type=bool, default=False)
    parser.add_argument('--use_bench_boost', type=bool, default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    
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
        output_path=args.output_path,
        budget=args.budget,
        max_players=args.max_players,
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        train_seasons=args.train_seasons,
        val_season=args.val_season,
        test_season=args.test_season,
        model=model_config,
        use_ilp=args.use_ilp,
        use_synergy=args.use_synergy,
        use_triple_captain=args.use_triple_captain,
        use_bench_boost=args.use_bench_boost
    )
    
    # 运行实验
    runner = ExperimentRunner(config)
    runner.run()

if __name__ == '__main__':
    main()