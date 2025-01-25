import os
import logging
from datetime import datetime
import json
import pandas as pd
from config import MODEL_SAVE_PATH

class FPLLogger:
    def __init__(self, name, log_dir=None):
        if log_dir is None:
            log_dir = os.path.join(MODEL_SAVE_PATH, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志格式
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 训练历史记录
        self.history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'points': [],
            'team_values': []
        }
        
        self.log_file = log_file
    
    def log_episode(self, episode, reward, loss, points, team_value):
        """记录每个episode的信息"""
        self.history['episodes'].append(episode)
        self.history['rewards'].append(reward)
        self.history['losses'].append(loss)
        self.history['points'].append(points)
        self.history['team_values'].append(team_value)
        
        self.logger.info(
            f"Episode {episode} - "
            f"Reward: {reward:.2f}, Loss: {loss:.4f}, "
            f"Points: {points:.1f}, Team Value: {team_value:.1f}"
        )
    
    def log_evaluation(self, results):
        """记录评估结果"""
        self.logger.info("\nEvaluation Results:")
        self.logger.info(f"Average Points: {results['avg_points']:.1f} ± {results['std_points']:.1f}")
        self.logger.info(f"Average Team Value: £{results['avg_value']:.1f}m")
        self.logger.info("\nFormation Usage:")
        self.logger.info(results['formations'].mean())
        
        # 保存详细结果
        results_file = self.log_file.replace('.log', '_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'avg_points': float(results['avg_points']),
                'std_points': float(results['std_points']),
                'avg_value': float(results['avg_value']),
                'formations': results['formations'].mean().to_dict(),
                'top_players': results['player_selections'].head(10).to_dict()
            }, f, indent=4)
    
    def save_history(self):
        """保存训练历史"""
        history_df = pd.DataFrame(self.history)
        history_file = self.log_file.replace('.log', '_history.csv')
        history_df.to_csv(history_file, index=False)
        self.logger.info(f"Training history saved to {history_file}")