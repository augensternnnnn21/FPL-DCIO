import os
import torch

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 路径配置
BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(BASE_DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DATA_PATH, "processed")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")

# 确保目录存在
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 游戏参数
BUDGET = 100.0
NUM_PLAYERS = 600  # 估计值，实际会根据数据调整

# 训练参数
TOTAL_EPISODES = 1000
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 5000

# 学习参数
META_LR = 0.0001          # 元学习率
INNER_LR = 0.001          # 内循环学习率
META_BATCH_SIZE = 4       # 元批次大小
INNER_STEPS = 5           # 内循环步数
GAMMA = 0.99              # 折扣因子
TAU = 0.005              # 目标网络软更新系数

# 探索参数
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# 网络参数
HIDDEN_DIM = 256
POSITION_EMBED_DIM = 32
NUM_ATTENTION_HEADS = 4
DROPOUT_RATE = 0.1

# 奖励权重
REWARD_WEIGHTS = {
    'points': 0.7,        # 得分权重
    'formation': 0.1,     # 阵型权重
    'budget': 0.1,        # 预算权重
    'stability': 0.05,    # 稳定性权重
    'potential': 0.05     # 潜力权重
}

# 特征配置
PLAYER_FEATURES = [
    'recent_points_mean',
    'recent_points_std',
    'form',
    'fixture_difficulty',
    'price_value',
    'team_strength',
    'position_scarcity',
    'value',
    'total_points',
    'is_home',
    'opponent_defense_strength'
]

# 位置要求
FORMATION_REQUIREMENTS = {
    'GK': 2,
    'DEF': 5,
    'MID': 5,
    'FWD': 3
}

# 保存和日志
SAVE_INTERVAL = 100       # 模型保存间隔
LOG_INTERVAL = 10         # 日志打印间隔
EVALUATE_INTERVAL = 50    # 评估间隔

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'