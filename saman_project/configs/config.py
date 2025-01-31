from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # 策略生成器配置
    strategy_hidden_dim: int = 256
    strategy_num_layers: int = 2
    strategy_dropout: float = 0.1
    max_seq_length: int = 50
    
    # 自适应信息模型配置
    info_hidden_dim: int = 128
    info_num_heads: int = 4
    info_dropout: float = 0.1
    
    # 元策略网络配置
    meta_hidden_dims: list = (512, 256, 128)
    meta_dropout: float = 0.2
    
@dataclass
class TrainingConfig:
    # 基础训练参数
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # 学习率调度
    lr_scheduler_factor: float = 0.1
    lr_scheduler_patience: int = 10
    
    # 损失函数权重
    ce_loss_weight: float = 1.0
    info_reg_weight: float = 0.1
    
    # 设备配置
    device: str = "cpu"
    num_workers: int = 4
    
    # 早停
    early_stopping_patience: int = 15
    
@dataclass
class SAMANConfig:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    # 数据配置
    data_path: str = "saman_project/data/"
    save_path: str = "saman_project/checkpoints/"
    
    # 特征配置
    feature_dims: dict = None
    num_strategies: int = 10  # 策略数量
    
    def __post_init__(self):
        self.feature_dims = {
            "player": 20,  # 球员特征维度
            "match": 15,   # 比赛特征维度
            "team": 10,    # 球队特征维度
            "meta": 5      # 元信息维度
        } 