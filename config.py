DEFAULT_CONFIG = ExperimentConfig(
    learning_rate=1e-5,  # 降低学习率
    gamma=0.95,          # 调整折扣因子
    num_episodes=5000,   # 增加训练轮次
    batch_size=32,       # 添加批次大小
    reward_scale=0.1     # 添加奖励缩放系数
) 