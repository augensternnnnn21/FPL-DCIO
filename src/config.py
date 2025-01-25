import os
import torch
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("\nDEBUG: Initial PROJECT_ROOT:", PROJECT_ROOT)

# 确保导出所有需要的类和变量
__all__ = ['PROJECT_ROOT', 'ExperimentConfig', 'ModelConfig', 'TrainingConfig', 'EnvironmentConfig', 'DEFAULT_CONFIG', 'COVID_CONFIG']

@dataclass
class ModelConfig:
    """模型配置"""
    input_dim: int = 12
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    temporal_window: int = 5
    max_players: int = 600

@dataclass
class TrainingConfig:
    num_episodes: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.0001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    gamma: float = 0.99
    tau: float = 0.005
    grad_clip: float = 1.0

@dataclass
class EnvironmentConfig:
    budget: float = 100.0
    max_players: int = 15
    history_length: int = 5

@dataclass
class GeneralizationConfig:
    num_test_episodes: int = 10
    perturbation_ratio: float = 0.1
    adaptation_steps: int = 100
    domain_lambda: float = 1.0

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 设备配置
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 随机种子
    random_seed: int = 42
    
    # 实验运行次数
    num_runs: int = 5
    
    # 数据配置
    train_seasons: List[str] = field(default_factory=lambda: [
        '2016-17', '2017-18', '2018-19', '2019-20', '2020-21'
    ])
    val_season: str = '2021-22'
    test_season: str = '2022-23'
    
    # 路径配置
    output_path: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
    )
    
    # 训练配置
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 模型配置
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # 泛化配置
    generalization: GeneralizationConfig = field(default_factory=GeneralizationConfig)
    
    # 日志配置
    log_interval: int = 10
    save_interval: int = 100
    evaluate_interval: int = 50
    
    def __post_init__(self):
        """初始化后的处理"""
        print("\nDEBUG: ExperimentConfig post_init")
        print(f"self.__dict__: {self.__dict__}")
        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)

def get_default_config() -> ExperimentConfig:
    """获取默认配置"""
    return ExperimentConfig()

# 为了保持向后兼容
DEFAULT_CONFIG = get_default_config()

# 创建 COVID 配置
print("\nDEBUG: Creating COVID_CONFIG...")
COVID_CONFIG = ExperimentConfig(
    train_seasons=['2019-20', '2020-21'],
    val_season='2021-22',
    test_season='2022-23'
)
print("COVID_CONFIG created, id:", id(COVID_CONFIG))
print("COVID_CONFIG dict:", COVID_CONFIG.__dict__)

# 训练配置
TRAINING_CONFIG = {
    'num_episodes': 1000,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'gamma': 0.99,
    'tau': 0.001,
    'update_every': 4,
    'buffer_size': int(1e5),
    'min_buffer_size': 1000,
    'gradient_clip': 1.0
}

# 模型配置
MODEL_CONFIG = {
    'input_dim': 12,
    'hidden_dim': 128,
    'num_heads': 4,
    'num_layers': 2,
    'dropout': 0.1,
    'activation': 'relu'
}

# 环境配置
ENV_CONFIG = {
    'budget': 100.0,
    'max_players': 15,
    'max_per_team': 3,
    'position_limits': {
        'GK': 2,
        'DEF': 5,
        'MID': 5,
        'FWD': 3
    }
}