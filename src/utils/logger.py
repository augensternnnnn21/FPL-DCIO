import logging
from pathlib import Path
import datetime
import json
import numpy as np
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class FPLLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建本次运行的时间戳目录
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / self.timestamp
        self.run_dir.mkdir(exist_ok=True)
        
        # 设置日志文件
        log_file = self.run_dir / "training.log"
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self._logger = logging.getLogger("FPL-DCIO")
        
        # 初始化指标记录
        self.metrics_history = {
            'rewards': [],
            'avg_rewards': [],
            'losses': [],
            'violations': [],
            'team_values': [],
            'points': [],
            'episodes': [],
            'steps': [],
            'total_reward': [],
            'avg_reward': [],
            'epsilon': [],
            'team_value': []
        }
        
        # 创建CSV文件记录详细指标
        self.metrics_file = self.run_dir / "metrics.csv"
        self.metrics_df = pd.DataFrame()
        
        # 打印训练头信息
        header = (
            "\n[训练日志] 项目名称: FPL阵容优化模型\n"
            "模型架构: DynamicSpatioTemporalMetaLearner\n"
            f"训练设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
            f"训练日期: {datetime.datetime.now().strftime('%Y-%m-%d')}\n"
            "\n超参数配置:\n"
            "  - 学习率: 0.001\n"
            "  - 折扣因子 (gamma): 0.99\n"
            "  - 探索率 (epsilon): 0.5 -> 0.05 (指数衰减)\n"
            "  - 批次大小 (batch_size): 32\n"
            "  - 回放缓冲区大小: 100000\n"
            "  - 目标网络更新频率: 0.005\n"
        )
        self.info(header)
        
        # 打印表头
        table_header = (
            "\nEpisode | 总步数 | 累计奖励 | 平均奖励 | 探索率 | 损失 (Policy/Value/Total) | 约束违反率 | 球队价值 | 得分\n"
            "---------|--------|----------|----------|--------|------------------------|------------|----------|------"
        )
        self.info(table_header)
    
    def info(self, message: str):
        """记录信息级别的日志"""
        self._logger.info(message)
    
    def error(self, message: str):
        """记录错误级别的日志"""
        self._logger.error(message)
    
    def warning(self, message: str):
        """记录警告级别的日志"""
        self._logger.warning(message)
    
    def debug(self, message: str):
        """记录调试级别的日志"""
        self._logger.debug(message)

    def log_training(self, episode, metrics):
        """记录训练指标"""
        try:
            # 更新历史记录
            self.metrics_history['episodes'].append(episode)
            self.metrics_history['steps'].append(metrics['steps'])
            self.metrics_history['total_reward'].append(metrics['total_reward'])
            self.metrics_history['avg_reward'].append(metrics['avg_reward'])
            self.metrics_history['epsilon'].append(metrics['epsilon'])
            self.metrics_history['violations'].append(metrics.get('violations', 0))
            self.metrics_history['team_value'].append(metrics.get('team_value', 0))
            self.metrics_history['points'].append(metrics.get('points', 0))
            
            # 打印当前episode的指标
            print(f"\rEpisode {episode:4d} | "
                  f"Steps: {metrics['steps']:5d} | "
                  f"Total Reward: {metrics['total_reward']:8.2f} | "
                  f"Avg Reward: {metrics['avg_reward']:6.2f} | "
                  f"Epsilon: {metrics['epsilon']:.3f} | "
                  f"Violations: {metrics.get('violations', 0)} | "
                  f"Team Value: {metrics.get('team_value', 0):.1f} | "
                  f"Points: {metrics.get('points', 0)}")
            
            # 每100轮保存一次数据和绘制图表
            if episode % 100 == 0:
                # 添加性能统计
                avg_reward = np.mean(self.metrics_history['total_reward'][-100:])
                avg_points = np.mean(self.metrics_history['points'][-100:])
                avg_violations = np.mean(self.metrics_history['violations'][-100:])
                
                stats_msg = (
                    f"\n{'='*50}\n"
                    f"Performance Statistics (Last 100 Episodes):\n"
                    f"Average Reward: {avg_reward:.2f}\n"
                    f"Average Points: {avg_points:.2f}\n"
                    f"Average Violations: {avg_violations:.2f}\n"
                    f"{'='*50}"
                )
                self.info(stats_msg)
                
                # 保存指标到CSV
                df = pd.DataFrame(self.metrics_history)
                df.to_csv(self.metrics_file, index=False)
                
                # 绘制训练曲线
                self._plot_training_curves()
            
        except Exception as e:
            self.error(f"Error in log_training: {str(e)}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 奖励曲线
        ax = axes[0, 0]
        ax.plot(self.metrics_history['rewards'], alpha=0.3, label='Reward')
        ax.plot(self.metrics_history['avg_rewards'], label='Avg Reward')
        ax.set_title('Rewards over Episodes')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        
        # 2. 损失曲线
        ax = axes[0, 1]
        ax.plot(self.metrics_history['losses'])
        ax.set_title('Loss over Episodes')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        
        # 3. 违规次数
        ax = axes[1, 0]
        ax.plot(self.metrics_history['violations'])
        ax.set_title('Constraint Violations')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Violations')
        
        # 4. 球队价值和得分
        ax = axes[1, 1]
        ax2 = ax.twinx()
        l1 = ax.plot(self.metrics_history['team_values'], 
                    color='blue', label='Team Value')
        l2 = ax2.plot(self.metrics_history['points'],
                     color='red', label='Points')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Team Value (£M)', color='blue')
        ax2.set_ylabel('Points', color='red')
        
        # 合并图例
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / f'training_curves.png')
        plt.close()
    
    def log_evaluation(self, season, metrics):
        """记录评估结果"""
        self.info(
            f"\nEvaluation on {season}:\n"
            f"Average Points: {metrics['avg_points']:.2f}\n"
            f"Total Value: {metrics['total_value']:.1f}M\n"
            f"Constraint Violations: {metrics['violations']}\n"
            f"Team Diversity: {metrics['team_diversity']:.2f}\n"
            f"Synergy Score: {metrics.get('synergy_score', 'N/A')}"
        )
    
    def log_team_selection(self, gameweek, selected_team):
        """记录球队选择"""
        self.info(f"\nGameweek {gameweek} Team Selection:")
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            players = selected_team[selected_team['position'] == pos]
            self.info(f"\n{pos}:")
            for _, player in players.iterrows():
                self.info(
                    f"  {player['name']} ({player['team']}) - "
                    f"£{player['value']}M, Points: {player['points']}"
                )