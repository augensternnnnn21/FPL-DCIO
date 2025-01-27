import os
import torch
import torch.nn.functional as F
import numpy as np
from environment import FPLEnv
from models import DCIO
from replay_buffer import ReplayBuffer
from config import *
import time
from datetime import datetime

def print_progress(episode, total_episodes, reward, epsilon, loss, avg_reward):
    """打印训练进度"""
    print(f"\rEpisode: {episode}/{total_episodes} | "
          f"Reward: {reward:.1f} | "
          f"Epsilon: {epsilon:.3f} | "
          f"Loss: {loss:.3f} | "
          f"Avg Reward: {avg_reward:.1f}", end="")

def print_team_info(env, action, total_reward=None, epsilon=None, episode=None):
    """打印球队阵容信息"""
    selected_players = env.data.iloc[action]
    positions = selected_players['position'].value_counts()
    total_cost = selected_players['value'].sum()
    
    if episode is not None:
        print(f"\nEpisode {episode} 选择结果:")
    print("\n当前阵容:")
    print(f"位置分布: {dict(positions)}")
    print(f"总成本: {total_cost:.1f}")
    print("球员列表:")
    for _, player in selected_players.iterrows():
        print(f"{player['position']}: {player['name']} (£{player['value']}m)")
    if total_reward is not None:
        print(f"奖励: {total_reward:.1f}")
    if epsilon is not None:
        print(f"探索率: {epsilon:.3f}")
    print("-" * 50)

# 训练参数
TOTAL_EPISODES = 1000
PRINT_INTERVAL = 10
SAVE_INTERVAL = 100
MAX_GRAD_NORM = 1.0
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.99
TAU = 0.005
REPLAY_BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 10

def select_action(model, state, epsilon, env):
    """改进的动作选择函数"""
    if np.random.rand() < epsilon:
        # 随机探索时严格满足约束
        return _random_action_with_constraints(env)
    else:
        # 模型预测时应用位置掩码和预算约束
        return _model_based_action(model, state, env)

def _random_action_with_constraints(env):
    """带约束的随机动作选择"""
    action = []
    budget_remaining = BUDGET
    pos_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    selected_indices = set()
    
    for pos, required_count in pos_requirements.items():
        # 获取该位置所有可用且未选择的球员
        candidates = [
            i for i in range(env.num_players)
            if (env.data['position'].iloc[i] == pos) and 
               (i not in selected_indices) and
               (env.data['value'].iloc[i] <= budget_remaining)
        ]
        
        if len(candidates) < required_count:
            return _fallback_action(env)
            
        # 随机选择指定数量的球员
        selected = np.random.choice(candidates, size=required_count, replace=False)
        
        # 更新预算和已选球员
        for idx in selected:
            cost = env.data['value'].iloc[idx]
            if cost > budget_remaining:
                return _fallback_action(env)
            action.append(idx)
            selected_indices.add(idx)
            budget_remaining -= cost
    
    return np.array(action)

# train.py
def adjust_epsilon(episode):
    if episode < 300:  # 初期高探索
        return max(0.5, EPSILON_START * (EPSILON_DECAY ** episode))
    elif episode < 700:  # 中期逐步收敛
        return max(0.1, 0.5 * (0.9 ** (episode-300)))
    else:  # 后期低探索
        return EPSILON_END



def _model_based_action(model, state, env):
    """基于模型的动作选择，严格遵守约束"""
    with torch.no_grad():
        q_values = model(state)  # [1, num_players]
        action = []
        budget_remaining = BUDGET
        pos_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        selected_indices = set()
        
        # 获取当前比赛周可用的球员
        current_gw_data = env.data[env.data['gw'] == env.current_gw]
        available_players = set(current_gw_data['name'])
        
        # 按位置选择球员
        for pos, required_count in pos_requirements.items():
            candidates = []
            for i in range(env.num_players):
                player_name = env.inverse_id_map.get(i)
                if (i not in selected_indices and  # 未被选择
                    player_name in available_players and  # 当前比赛周可用
                    env.data['position'].iloc[i] == pos and  # 位置匹配
                    env.data['value'].iloc[i] <= budget_remaining):  # 预算充足
                    candidates.append(i)
            
            if len(candidates) < required_count:
                return _fallback_action(env)
            
            # 获取候选人的Q值
            candidate_q = q_values.squeeze()[candidates]
            sorted_indices = torch.argsort(candidate_q, descending=True)
            
            # 选择Q值最高的球员
            selected = []
            for idx in sorted_indices:
                candidate_idx = candidates[idx.item()]
                cost = env.data['value'].iloc[candidate_idx]
                
                if len(selected) == required_count:
                    break
                    
                if cost <= budget_remaining:
                    selected.append(candidate_idx)
                    selected_indices.add(candidate_idx)
                    budget_remaining -= cost
            
            if len(selected) < required_count:
                return _fallback_action(env)
            
            action.extend(selected)
        
        if len(action) != 15:
            return _fallback_action(env)
        
        # 最终验证
        total_cost = sum(env.data['value'].iloc[i] for i in action)
        if total_cost > BUDGET:
            return _fallback_action(env)
            
        return np.array(action)

def _fallback_action(env):
    """保底策略：严格按预算和位置要求选择最便宜的球员"""
    action = []
    budget_remaining = BUDGET
    pos_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    selected_indices = set()
    
    for pos, required_count in pos_requirements.items():
        candidates = [
            i for i in range(env.num_players)
            if (env.data['position'].iloc[i] == pos) and 
               (i not in selected_indices) and
               (env.data['value'].iloc[i] <= budget_remaining)
        ]
        
        if not candidates:
            return np.random.choice(env.num_players, size=15, replace=False)
            
        candidates.sort(key=lambda x: env.data['value'].iloc[x])
        
        for _ in range(required_count):
            if not candidates:
                return np.random.choice(env.num_players, size=15, replace=False)
                
            idx = candidates.pop(0)
            cost = env.data['value'].iloc[idx]
            
            if cost > budget_remaining:
                return np.random.choice(env.num_players, size=15, replace=False)
                
            action.append(idx)
            selected_indices.add(idx)
            budget_remaining -= cost
    
    if len(action) != 15:
        return np.random.choice(env.num_players, size=15, replace=False)
    
    return np.array(action)

def train(season="2016-17"):
    env = FPLEnv(season=season, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建增强型邻接矩阵
    adj = torch.zeros(NUM_PLAYERS, NUM_PLAYERS)
    team_map = env.data.groupby('name')['team'].first().to_dict()
    pos_map = env.data.groupby('name')['position'].first().to_dict()
    values = env.data.groupby('name')['value'].first().to_dict()
    
    for i in range(NUM_PLAYERS):
        for j in range(i+1, NUM_PLAYERS):
            same_team = team_map[env.data['name'].iloc[i]] == team_map[env.data['name'].iloc[j]]
            same_pos = pos_map[env.data['name'].iloc[i]] == pos_map[env.data['name'].iloc[j]]
            price_diff = abs(values[env.data['name'].iloc[i]] - values[env.data['name'].iloc[j]]) < 2.0
            if (same_team or same_pos) and price_diff:
                adj[i,j] = adj[j,i] = 1

    model = DCIO(adj.to(device), env.data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    # 训练循环
    rewards_history = []
    
    # 初始化epsilon
    epsilon = EPSILON_START
    
    # 预填充经验池
    state, _ = env.reset()
    state = torch.FloatTensor(state).to(device)
    while len(replay_buffer) < MIN_REPLAY_SIZE:
        action = select_action(model, state, epsilon, env)  # 使用初始epsilon
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).to(device)
        replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
        state = next_state if not done else torch.FloatTensor(env.reset()[0]).to(device)

    # 主训练循环
    for episode in range(TOTAL_EPISODES):
        epsilon = adjust_epsilon(episode)
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)
        total_reward = 0
        episode_loss = 0
        
        while True:
            action = select_action(model, state, epsilon, env)
            
            # 每10个episode打印一次详细信息
            if episode % 10 == 0:
                print_team_info(env, action, epsilon=epsilon, episode=episode)
                
            next_state, reward, done, _, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).to(device)
            replay_buffer.push(state.cpu().numpy(), action, reward, next_state, done)
            
            if len(replay_buffer) >= BATCH_SIZE:
                # 采样并训练
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = [t.to(device) for t in batch]
                
                # 计算目标Q值
                with torch.no_grad():
                    next_q = model(next_states, use_target=True)  # [batch_size, num_players]
                    
                    # 获取每个状态的最优动作的Q值
                    next_q_max = []
                    for i in range(BATCH_SIZE):
                        ns = next_states[i].unsqueeze(0)
                        next_action = select_action(model, ns, 0, env)  # 获取最优动作
                        # 计算该动作的Q值
                        action_q = next_q[i][next_action].mean()  # 使用选中球员的平均Q值
                        next_q_max.append(action_q)
                    
                    next_q_max = torch.stack(next_q_max)
                    targets = rewards + GAMMA * next_q_max * (1 - dones)
                
                # 计算当前Q值
                current_q = []
                q_values = model(states)  # [batch_size, num_players]
                for i in range(BATCH_SIZE):
                    action_indices = actions[i]
                    current_q.append(q_values[i][action_indices].mean())  # 使用选中球员的平均Q值
                current_q = torch.stack(current_q)
                
                # 计算损失并更新
                loss = F.mse_loss(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                episode_loss += loss.item()
            
            total_reward += reward
            state = next_state_tensor
            
            if done:
                break
        
        # 更新探索率
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # 记录并保存
        rewards_history.append(total_reward)
        if (episode+1) % SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"dcio_{season}_ep{episode+1}.pth"))
            # 在保存模型时打印详细信息
            print_team_info(env, action, total_reward, epsilon, episode)
        else:
            print(f"Episode {episode+1} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")

    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"dcio_{season}_final.pth"))
    return rewards_history

if __name__ == "__main__":
    train()