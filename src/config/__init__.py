import os
import torch

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 路径配置
BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(BASE_DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DATA_PATH, "processed")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")

# 确保目录存在
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 模型配置
NUM_PLAYERS = 600
HIDDEN_DIM = 64
NUM_ATTENTION_HEADS = 4
DROPOUT_RATE = 0.1

# 训练配置
BUDGET = 100.0
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# 位置要求
FORMATION_REQUIREMENTS = {
    'GK': 2,
    'DEF': 5,
    'MID': 5,
    'FWD': 3
}

# 保存和日志
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
EVALUATE_INTERVAL = 50

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 导入实验配置
from .experiment_config import ExperimentConfig, DEFAULT_CONFIG, COVID_CONFIG

__all__ = [
    # 路径配置
    'PROJECT_ROOT',
    'BASE_DATA_PATH',
    'RAW_DATA_PATH',
    'PROCESSED_DATA_PATH',
    'MODEL_SAVE_PATH',
    
    # 模型配置
    'NUM_PLAYERS',
    'HIDDEN_DIM',
    'NUM_ATTENTION_HEADS',
    'DROPOUT_RATE',
    
    # 训练配置
    'BUDGET',
    'EPSILON_START',
    'EPSILON_END',
    'EPSILON_DECAY',
    
    # 位置要求
    'FORMATION_REQUIREMENTS',
    
    # 保存和日志
    'SAVE_INTERVAL',
    'LOG_INTERVAL',
    'EVALUATE_INTERVAL',
    
    # 设备配置
    'DEVICE',
    
    # 实验配置
    'ExperimentConfig',
    'DEFAULT_CONFIG',
    'COVID_CONFIG'
]