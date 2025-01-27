import os
import torch
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any, Optional
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
    # 数据路径
    data_path: str
    output_path: str
    
    # 环境参数
    budget: float = 100.0
    max_players: int = 15
    
    # 训练参数
    num_episodes: int = 1000
    learning_rate: float = 0.001
    gamma: float = 0.99
    batch_size: int = 32          # 批次大小
    memory_size: int = 100000     # 记忆缓冲区大小
    min_memory_size: int = 1000  # 添加最小记忆大小
    target_update: int = 10       # 目标网络更新频率
    
    # 探索参数
    initial_epsilon: float = 1.0    # 初始探索率 (原 epsilon_start)
    epsilon_end: float = 0.01     # 最终探索率
    epsilon_decay: float = 0.995  # 探索率衰减
    min_epsilon: float = 0.01     # 最小探索率
    
    # 训练数据配置
    train_seasons: List[str] = None
    val_season: str = None
    test_season: str = None
    
    # 模型配置
    model: ModelConfig = ModelConfig()
    
    # 特性开关
    use_ilp: bool = True
    use_synergy: bool = True
    use_triple_captain: bool = False
    use_bench_boost: bool = False
    
    # 设备配置
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __post_init__(self):
        """初始化后的处理"""
        if self.train_seasons is None:
            self.train_seasons = ['2019-20', '2020-21']
        if self.val_season is None:
            self.val_season = '2021-22'
        if self.test_season is None:
            self.test_season = '2022-23'

    def to_dict(self):
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'output_path': self.output_path,
            'budget': self.budget,
            'max_players': self.max_players,
            'num_episodes': self.num_episodes,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size,
            'min_memory_size': self.min_memory_size,
            'target_update': self.target_update,
            'initial_epsilon': self.initial_epsilon,  # 更新名称
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'train_seasons': self.train_seasons,
            'val_season': self.val_season,
            'test_season': self.test_season,
            'model': self.model.__dict__,
            'use_ilp': self.use_ilp,
            'use_synergy': self.use_synergy,
            'use_triple_captain': self.use_triple_captain,
            'use_bench_boost': self.use_bench_boost,
            'device': self.device
        }

def get_default_config() -> ExperimentConfig:
    """获取默认配置"""
    return ExperimentConfig(
        data_path=os.path.join(PROJECT_ROOT, 'data'),
        output_path=os.path.join(PROJECT_ROOT, 'experiments'),  # 添加默认输出路径
        train_seasons=['2016-17', '2017-18', '2018-19'],
        val_season='2019-20',
        test_season='2020-21',
        budget=BUDGET,
        max_players=NUM_PLAYERS,
        model=ModelConfig()  # 使用默认模型配置
    )

# 为了保持向后兼容
DEFAULT_CONFIG = get_default_config()

# 创建 COVID 配置
print("\nDEBUG: Creating COVID_CONFIG...")
COVID_CONFIG = ExperimentConfig(
    data_path=os.path.join(PROJECT_ROOT, 'data'),
    output_path=os.path.join(PROJECT_ROOT, 'experiments'),  # 添加输出路径
    train_seasons=['2019-20', '2020-21'],
    val_season='2021-22',
    test_season='2022-23',
    budget=BUDGET,
    max_players=NUM_PLAYERS,
    model=ModelConfig()  # 使用默认模型配置
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

@dataclass
class LoggingConfig:
    log_interval: int = 10  # 每多少轮记录一次训练信息
    eval_interval: int = 100  # 每多少轮评估一次
    save_interval: int = 1000  # 每多少轮保存一次模型
    plot_interval: int = 100  # 每多少轮绘制一次图表
    verbose: bool = True  # 是否输出详细信息