import logging
from pathlib import Path
import datetime
import json

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
        self.logger = logging.getLogger("FPL-DCIO")
    
    def log_training(self, episode, metrics):
        """记录训练信息"""
        self.logger.info(
            f"Episode {episode} | "
            f"Reward: {metrics['reward']:.2f} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Team Value: {metrics['team_value']:.1f}M | "
            f"Points: {metrics['points']} | "
            f"Violations: {metrics['violations']}"
        )
        
        # 保存详细指标到JSON
        metrics_file = self.run_dir / "metrics.json"
        with open(metrics_file, "a") as f:
            json.dump({
                "episode": episode,
                **metrics
            }, f)
            f.write("\n")
    
    def log_evaluation(self, season, metrics):
        """记录评估结果"""
        self.logger.info(
            f"\nEvaluation on {season}:\n"
            f"Average Points: {metrics['avg_points']:.2f}\n"
            f"Total Value: {metrics['total_value']:.1f}M\n"
            f"Constraint Violations: {metrics['violations']}\n"
            f"Team Diversity: {metrics['team_diversity']:.2f}\n"
            f"Synergy Score: {metrics.get('synergy_score', 'N/A')}"
        )
    
    def log_team_selection(self, gameweek, selected_team):
        """记录球队选择"""
        self.logger.info(f"\nGameweek {gameweek} Team Selection:")
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            players = selected_team[selected_team['position'] == pos]
            self.logger.info(f"\n{pos}:")
            for _, player in players.iterrows():
                self.logger.info(
                    f"  {player['name']} ({player['team']}) - "
                    f"£{player['value']}M, Points: {player['points']}"
                )