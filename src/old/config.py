import os

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 路径配置 - 修正数据目录路径
BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "..", "data")  # 回退一级到项目根目录
PROCESSED_DATA_PATH = os.path.join(BASE_DATA_PATH, "processed")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")

# 确保目录存在
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 训练参数
NUM_PLAYERS = 500       # 每个赛季最多考虑的球员数量
BUDGET = 100            # 球队预算（单位：百万）
HIDDEN_DIM = 128         # GNN隐藏层维度
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_DECAY = 0.98

# 阵容要求
FORMATION_REQUIREMENTS = {
    'GK': (2, 2),  # (最少, 最多)
    'DEF': (5, 5),
    'MID': (5, 5),
    'FWD': (3, 3)
}

# 游戏参数
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
TAU = 0.1
REPLAY_BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1