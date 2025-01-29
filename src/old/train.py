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
        
        # 按位置选择球员
        for pos, required_count in pos_requirements.items():
            # 获取该位置所有合法候选人
            candidates = []
            for i in range(env.num_players):
                if (i not in selected_indices and  # 未被选择
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

def train_sequential(seasons=TRAIN_SEASONS):
    """按时间序列依次训练每个赛季"""
    print("开始序列训练...")
    
    # 初始化环境和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 记录每个赛季的训练结果
    all_rewards_history = {}
    
    # 添加早停相关参数
    EARLY_STOP_PATIENCE = 50  # 如果50个episode没有提升就早停
    EARLY_STOP_THRESHOLD = 0.01  # 提升阈值
    
    # 按顺序训练每个赛季
    for season_idx, season in enumerate(seasons):
        print(f"\n开始训练赛季 {season} ({season_idx+1}/{len(seasons)})")
        
        # 初始化环境
        env = FPLEnv(season=season, verbose=False)
        
        # 构建邻接矩阵
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
        
        # 初始化或加载模型
        if season_idx == 0:
            model = DCIO(adj.to(device), env.data).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        else:
            # 加载上一赛季的模型权重
            prev_season = seasons[season_idx-1]
            model_path = os.path.join(MODEL_SAVE_PATH, f"dcio_{prev_season}_final.pth")
            model.load_state_dict(torch.load(model_path))
            # 使用较小的学习率进行迁移学习
            optimizer = torch.optim.Adam(model.parameters(), lr=TRANSFER_LEARNING_RATE)
        
        # 训练当前赛季
        replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        rewards_history = []
        epsilon = EPSILON_START
        
        # 早停相关变量
        best_avg_reward = float('-inf')
        patience_counter = 0
        best_model_state = None
        
        # 添加奖励缩放
        REWARD_SCALE = 0.01  # 缩放奖励值
        
        # 预填充经验池时就应用奖励缩放
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)
        
        print("预填充经验池...")
        while len(replay_buffer) < MIN_REPLAY_SIZE:
            action = select_action(model, state, epsilon, env)
            next_state, reward, done, _, _ = env.step(action)
            # 缩放奖励
            reward = reward * REWARD_SCALE
            next_state = torch.FloatTensor(next_state).to(device)
            replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
            state = next_state if not done else torch.FloatTensor(env.reset()[0]).to(device)
        
        print(f"开始训练 {EPISODES_PER_SEASON} 轮...")
        for episode in range(EPISODES_PER_SEASON):
            epsilon = adjust_epsilon(episode)
            state, _ = env.reset()
            state = torch.FloatTensor(state).to(device)
            total_reward = 0
            episode_loss = 0
            
            while True:
                action = select_action(model, state, epsilon, env)
                
                # 每50个episode打印详细信息
                if episode % 50 == 0:
                    print_team_info(env, action, epsilon=epsilon, episode=episode)
                    
                next_state, reward, done, _, _ = env.step(action)
                # 缩放奖励
                reward = reward * REWARD_SCALE
                next_state_tensor = torch.FloatTensor(next_state).to(device)
                replay_buffer.push(state.cpu().numpy(), action, reward, next_state, done)
                
                if len(replay_buffer) >= BATCH_SIZE:
                    batch = replay_buffer.sample(BATCH_SIZE)
                    states, actions, rewards, next_states, dones = [t.to(device) for t in batch]
                    
                    # 计算目标Q值时添加值域限制
                    with torch.no_grad():
                        next_q = model(next_states, use_target=True)
                        next_q = torch.clamp(next_q, -100, 100)  # 限制Q值范围
                        
                        next_q_max = []
                        for i in range(BATCH_SIZE):
                            ns = next_states[i].unsqueeze(0)
                            next_action = select_action(model, ns, 0, env)
                            action_q = next_q[i][next_action].mean()
                            next_q_max.append(action_q)
                        
                        next_q_max = torch.stack(next_q_max)
                        targets = rewards + GAMMA * next_q_max * (1 - dones)
                        targets = torch.clamp(targets, -100, 100)  # 限制目标值范围
                    
                    # 计算当前Q值时也添加值域限制
                    current_q = []
                    q_values = model(states)
                    q_values = torch.clamp(q_values, -100, 100)  # 限制Q值范围
                    
                    for i in range(BATCH_SIZE):
                        action_indices = actions[i]
                        current_q.append(q_values[i][action_indices].mean())
                    current_q = torch.stack(current_q)
                    
                    # 使用Huber Loss代替MSE Loss
                    loss = F.smooth_l1_loss(current_q, targets)
                    
                    # 添加梯度裁剪
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
            
            # 记录奖励
            rewards_history.append(total_reward)
            
            # 每轮结束后打印基本信息
            print(f"\rEpisode {episode+1}/{EPISODES_PER_SEASON} | "
                  f"Reward: {total_reward:.1f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Loss: {episode_loss:.3f}", end="")
            
            # 计算最近100个episode的平均奖励
            if len(rewards_history) >= 100:
                current_avg_reward = np.mean(rewards_history[-100:])
                
                # 检查是否有提升
                if current_avg_reward > best_avg_reward + EARLY_STOP_THRESHOLD:
                    best_avg_reward = current_avg_reward
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # 早停检查
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"\n\n触发早停! 最近{EARLY_STOP_PATIENCE}个episode没有显著提升")
                    print(f"最佳平均奖励: {best_avg_reward:.1f}")
                    model.load_state_dict(best_model_state)
                    break
            
            # 定期保存和打印详细信息
            if (episode+1) % SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), 
                         os.path.join(MODEL_SAVE_PATH, f"dcio_{season}_ep{episode+1}.pth"))
                print(f"\n\nEpisode {episode+1}/{EPISODES_PER_SEASON} 详细信息:")
                print(f"当前奖励: {total_reward:.1f}")
                print(f"探索率: {epsilon:.3f}")
                print(f"平均奖励 (last 100): {np.mean(rewards_history[-100:]):.1f}")
                print(f"最佳平均奖励: {best_avg_reward:.1f}")
                print(f"耐心计数器: {patience_counter}/{EARLY_STOP_PATIENCE}")
                print("-" * 50)
        
        # 保存最终模型
        torch.save(model.state_dict(), 
                  os.path.join(MODEL_SAVE_PATH, f"dcio_{season}_final.pth"))
        
        # 记录该赛季的训练历史
        all_rewards_history[season] = rewards_history
        
        # 打印该赛季的训练结果
        print(f"\n赛季 {season} 训练完成")
        print(f"最终平均奖励 (last 100): {np.mean(rewards_history[-100:]):.1f}")
        print(f"最高奖励: {max(rewards_history):.1f}")
        print(f"训练轮数: {episode+1}")
    
    return all_rewards_history

def train_epoch(model, optimizer, env, device, batch_size=32):
    """单个epoch的MAML训练"""
    model.train()
    total_loss = 0
    
    for _ in range(batch_size):
        # 1. 采样support set和query set
        support_state, _ = env.reset()
        query_state, _ = env.reset()
        
        support_x = torch.FloatTensor(support_state).to(device)
        query_x = torch.FloatTensor(query_state).to(device)
        
        # 使用实际得分作为目标
        support_y = torch.FloatTensor(env.data['total_points'].values).to(device)
        query_y = torch.FloatTensor(env.data['total_points'].values).to(device)
        
        # 2. 在support set上适应
        adapted_params = model.adapt(support_x, support_y)
        
        # 3. 在query set上评估和更新
        query_pred = model(query_x, adapted_params)
        loss = F.mse_loss(query_pred, query_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / batch_size

def train(model, env, device, num_epochs=100):
    """MAML训练主循环"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        loss = train_epoch(model, optimizer, env, device)
        scheduler.step()
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'models/dcio_maml.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

if __name__ == "__main__":
    rewards_history = train_sequential()