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
import random
import torch.optim as optim

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

def train():
    # 定义训练集和测试集赛季
    train_seasons = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
    test_season = "2023-24"
    
    print("初始化训练环境...")
    # 创建多个训练环境
    train_envs = []
    for season in train_seasons:
        try:
            env = FPLEnv(season=season, verbose=False)
            train_envs.append(env)
            print(f"成功加载 {season} 赛季数据")
        except Exception as e:
            print(f"警告: 加载 {season} 赛季数据失败: {str(e)}")
    
    if not train_envs:
        raise ValueError("没有成功加载任何训练赛季数据")
    
    # 使用第一个环境的状态维度来初始化模型
    state_dim = train_envs[0].observation_space.shape[0]
    action_dim = train_envs[0].action_space.n
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 初始化模型和优化器
    policy_net = DCIO(torch.zeros(NUM_PLAYERS, NUM_PLAYERS), train_envs[0].data).to(device)
    target_net = DCIO(torch.zeros(NUM_PLAYERS, NUM_PLAYERS), train_envs[0].data).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    # 训练循环
    total_steps = 0
    epsilon = EPSILON_START
    
    print("开始训练...")
    for episode in range(TOTAL_EPISODES):
        # 随机选择一个训练环境
        env = random.choice(train_envs)
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            # epsilon-greedy策略
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.max(1)[1].item()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # 存储经验
            memory.push(state, action, reward, next_state, done)
            state = next_state
            
            # 训练
            if len(memory) > MIN_REPLAY_SIZE:
                batch = memory.sample(BATCH_SIZE)
                loss = compute_td_loss(batch, policy_net, target_net)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
            
            total_steps += 1
            # 更新目标网络
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        # 衰减epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # 每N个episode评估一次
        if episode % EVAL_FREQUENCY == 0:
            eval_reward = evaluate(policy_net, test_season)
            print(f"Episode {episode}, 训练奖励: {episode_reward:.2f}, 测试奖励: {eval_reward:.2f}, Epsilon: {epsilon:.2f}")
            
            # 保存最佳模型
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(policy_net.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))
    
    print("训练完成!")
    return policy_net

def evaluate(model, test_season, num_episodes=5):
    """在测试赛季上评估模型"""
    model.eval()
    test_env = FPLEnv(season=test_season, verbose=False)
    total_rewards = []
    
    with torch.no_grad():
        for _ in range(num_episodes):
            state = test_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(state_tensor).max(1)[1].item()
                state, reward, done, _ = test_env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
    
    model.train()
    return np.mean(total_rewards)

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 添加一些训练参数
    NUM_EPISODES = 1000
    MAX_STEPS = 100
    EVAL_FREQUENCY = 10
    best_eval_reward = float('-inf')
    
    # 开始训练
    trained_model = train()