from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class TrainingConfig:
    """训练配置"""
    num_episodes: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    gamma: float = 0.99
    tau: float = 0.005  # 目标网络软更新参数
    grad_clip: float = 1.0

@dataclass
class ModelConfig:
    """模型配置"""
    input_dim: int = 12
    hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    temporal_window: int = 5
    max_players: int = 600

@dataclass
class GeneralizationConfig:
    """泛化性评估配置"""
    num_test_episodes: int = 10
    perturbation_ratio: float = 0.1
    adaptation_steps: int = 100
    domain_lambda: float = 1.0

@dataclass
class ExperimentConfig:
    """实验总配置"""
    # 基础设置
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_seed: int = 42
    num_runs: int = 5  # 每个实验重复次数
    
    # 数据配置
    train_seasons: List[str] = None
    val_season: str = "2021-22"
    test_season: str = "2022-23"
    
    # 组件配置
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    generalization: GeneralizationConfig = GeneralizationConfig()
    
    # 日志配置
    log_interval: int = 10
    save_interval: int = 100
    evaluate_interval: int = 50
    
    def __post_init__(self):
        if self.train_seasons is None:
            self.train_seasons = [
                "2016-17", "2017-18", "2018-19", 
                "2019-20", "2020-21"
            ]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'device': str(self.device),
            'random_seed': self.random_seed,
            'num_runs': self.num_runs,
            'train_seasons': self.train_seasons,
            'val_season': self.val_season,
            'test_season': self.test_season,
            'training': vars(self.training),
            'model': vars(self.model),
            'generalization': vars(self.generalization),
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'evaluate_interval': self.evaluate_interval
        }

# 默认配置
DEFAULT_CONFIG = ExperimentConfig()

# COVID实验配置
COVID_CONFIG = ExperimentConfig(
    train_seasons=["2016-17", "2017-18", "2018-19"],
    val_season="2019-20",  # COVID开始
    test_season="2020-21",  # COVID期间
    generalization=GeneralizationConfig(
        num_test_episodes=20,  # 增加测试轮数
        perturbation_ratio=0.15,  # 增加扰动强度
        adaptation_steps=150  # 增加适应步数
    )
)

# 跨赛季泛化配置
CROSS_SEASON_CONFIG = ExperimentConfig(
    train_seasons=["2016-17", "2017-18", "2018-19"],
    val_season="2020-21",
    test_season="2022-23",
    model=ModelConfig(
        hidden_dim=128,  # 增加模型容量
        num_heads=8  # 增加注意力头数
    ),
    training=TrainingConfig(
        num_episodes=2000,  # 增加训练轮数
        learning_rate=5e-5  # 降低学习率
    )
) 