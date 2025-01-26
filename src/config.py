import os
import torch
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any
import warnings

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("\nDEBUG: Initial PROJECT_ROOT:", PROJECT_ROOT)

# 确保导出所有需要的类和变量
__all__ = ['PROJECT_ROOT', 'ExperimentConfig', 'ModelConfig', 'TrainingConfig', 'EnvironmentConfig', 'DEFAULT_CONFIG', 'COVID_CONFIG']

# 添加或确认这些常量的定义
BUDGET = 100.0  # 预算限制
NUM_PLAYERS = 15  # 最大球员数量

@dataclass
class ModelConfig:
    """模型配置"""
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    bidirectional: bool = True
    use_attention: bool = True

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
    # 数据相关
    data_path: str
    train_seasons: List[str]
    val_season: str
    test_season: str
    
    # 环境相关
    budget: float = 100.0
    max_players: int = 15
    
    # 训练相关
    num_episodes: int = 1000
    learning_rate: float = 0.001
    gamma: float = 0.99
    num_runs: int = 5
    
    # 设备和输出
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path: str = 'experiments'
    random_seed: int = 42
    
    # 模型配置
    model: ModelConfig = ModelConfig()

    def __post_init__(self):
        """初始化后的处理"""
        # 确保路径格式正确
        self.data_path = str(self.data_path)
        self.output_path = str(self.output_path)
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数"""
        if self.budget <= 0:
            raise ValueError(f"预算必须为正数: {self.budget}")
        if self.max_players <= 0:
            raise ValueError(f"最大球员数必须为正数: {self.max_players}")
        if self.num_episodes <= 0:
            raise ValueError(f"训练回合数必须为正数: {self.num_episodes}")
        if not (0 <= self.gamma <= 1):
            raise ValueError(f"折扣因子必须在[0,1]范围内: {self.gamma}")

def get_default_config() -> ExperimentConfig:
    """获取默认配置"""
    return ExperimentConfig(
        data_path=os.path.join(PROJECT_ROOT, 'data'),  # 添加默认数据路径
        train_seasons=['2016-17', '2017-18', '2018-19'],
        val_season='2019-20',
        test_season='2020-21'
    )

# 为了保持向后兼容
DEFAULT_CONFIG = get_default_config()

# 创建 COVID 配置
print("\nDEBUG: Creating COVID_CONFIG...")
COVID_CONFIG = ExperimentConfig(
    data_path=os.path.join(PROJECT_ROOT, 'data'),  # 添加data_path
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