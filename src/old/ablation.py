# 新建文件 src/old/ablation.py

import torch
import torch.nn as nn
from models import DCIO, GNNLayer
from environment import FPLEnv
from train import train_sequential
from config import *

class DCIOWithoutGNN(DCIO):
    """移除GNN的版本"""
    def __init__(self, adj_matrix, player_data):
        super(DCIO, self).__init__()
        self.data = player_data
        self.num_players = len(player_data['name'].unique())
        self.input_dim = 5
        self.hidden_dim = HIDDEN_DIM
        
        # 只使用MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_PLAYERS)
        )
    
    def forward(self, state):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        return self.mlp(state)

class DCIOWithoutAttention(DCIO):
    """移除注意力机制的版本"""
    def __init__(self, adj_matrix, player_data):
        super().__init__(adj_matrix, player_data)
        # 使用简单的GNN层
        self.gnn1 = SimpleGNNLayer(self.hidden_dim)
        self.gnn2 = SimpleGNNLayer(self.hidden_dim)

def run_ablation_study():
    """运行消融实验"""
    variants = {
        "Full Model": DCIO,
        "Without GNN": DCIOWithoutGNN,
        "Without Attention": DCIOWithoutAttention
    }
    
    results = {}
    for name, model_class in variants.items():
        print(f"\n测试变体: {name}")
        rewards = train_sequential(model_class=model_class)
        results[name] = rewards
    
    # 分析结果
    print("\n消融实验结果:")
    for variant, rewards in results.items():
        for season, reward_history in rewards.items():
            print(f"\n{variant} - {season}:")
            print(f"最终平均奖励: {np.mean(reward_history[-100:]):.1f}")
            print(f"最高奖励: {max(reward_history):.1f}")
            print(f"收敛速度: {len(reward_history)} episodes")

if __name__ == "__main__":
    run_ablation_study()