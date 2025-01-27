import os
import torch
import numpy as np
from environment import FPLEnv
from models import DCIO
from config import *

# evaluate.py
def evaluate(season: str):
    print(f"\n开始评估 {season} 赛季...")
    env = FPLEnv(season=season, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 构建邻接矩阵（新增代码）
    adj = torch.zeros(NUM_PLAYERS, NUM_PLAYERS)
    team_map = env.data.groupby('name')['team'].first().to_dict()
    for i in range(NUM_PLAYERS):
        for j in range(i+1, NUM_PLAYERS):
            if team_map[env.data['name'].unique()[i]] == team_map[env.data['name'].unique()[j]]:
                adj[i,j] = adj[j,i] = 1
    
    print(f"邻接矩阵构建完成，形状: {adj.shape}")
    
    # 初始化模型时传递环境数据和邻接矩阵（修正参数顺序）
    model = DCIO(adj.to(device), env.data).to(device)  # 先传递 adj_matrix，再传递 player_data
    model.load_state_dict(torch.load("/Users/zhangyufei/Desktop/FPL-DCIO/models/dcio_2016-17_final.pth"))
    model.eval()
    print("模型加载完成")
    
    total_rewards = []
    valid_selections = 0
    total_cost_list = []
    formation_valid_count = 0
    
    print("\n开始评估...")
    for episode in range(10):
        print(f"\nEpisode {episode + 1}/10:")
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)
        episode_reward = 0
        
        while True:
            with torch.no_grad():
                q = model(state.unsqueeze(0))
                action = q.squeeze().topk(15).indices.cpu().numpy()
            
            # 打印选择的球员信息
            selected_players = env.data.iloc[action]
            total_cost = selected_players['value'].sum()
            positions = selected_players['position'].value_counts()
            
            print(f"\n选择的阵容:")
            print(f"位置分布: {dict(positions)}")
            print(f"总成本: {total_cost:.1f}")
            
            next_state, reward, done, _, _ = env.step(action)
            state = torch.FloatTensor(next_state).to(device)
            episode_reward += reward
            
            # 检查选择是否合法
            if reward > 0:
                valid_selections += 1
                total_cost_list.append(total_cost)
                if all(positions.get(pos, 0) == req for pos, req in env.formation_requirements.items()):
                    formation_valid_count += 1
            
            if done:
                break
        
        print(f"Episode {episode + 1} 奖励: {episode_reward:.1f}")
        total_rewards.append(episode_reward)
    
    print("\n评估结果:")
    print(f"平均奖励: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    print(f"合法选择比例: {valid_selections/10:.1%}")
    print(f"阵容合法比例: {formation_valid_count/10:.1%}")
    if total_cost_list:
        print(f"平均总成本: {np.mean(total_cost_list):.1f} ± {np.std(total_cost_list):.1f}")

if __name__ == "__main__":
    evaluate("2016-17")
    evaluate("2017-18")  # 跨赛季验证