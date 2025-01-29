import os
import random
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from utils.logger import FPLLogger
from utils.replay_buffer import ReplayBuffer

def process_state(state):
    """处理状态数据"""
    if isinstance(state, dict):
        # 连接所有特征
        features = [
            state['player_features'],
            state['position_onehot'],
            state['team_onehot']
        ]
        
        # 确保所有特征都是2D的 [num_players, features]
        features = [f.squeeze() if len(f.shape) > 2 else f for f in features]
        
        # 连接特征
        state_tensor = torch.cat([
            torch.FloatTensor(f) for f in features
        ], dim=1)
        
        # 添加batch维度 [1, num_players, total_features]
        return state_tensor.unsqueeze(0)
    else:
        return torch.FloatTensor(state).unsqueeze(0)

def select_action(q_values, formation_constraints):
    """选择动作"""
    # 获取每个球员的选择概率
    probs = F.softmax(q_values, dim=1)
    
    # 使用阈值选择动作
    threshold = 0.5
    action = (probs > threshold).float()
    
    # 确保至少选择了一个球员
    if action.sum() == 0:
        # 如果没有球员被选中，选择概率最高的
        max_indices = probs.argmax(dim=1)
        action[0, max_indices] = 1
    
    # 检查是否满足阵容约束
    if not (action.squeeze() == formation_constraints).all():
        # 如果阵容约束不满足，重新选择
        while not (action.squeeze() == formation_constraints).all():
            action = env.action_space.sample()
    
    return action

def compute_td_loss(model, target_model, batch, device):
    """计算TD损失"""
    states = torch.FloatTensor(batch.state).to(device)
    actions = torch.FloatTensor(batch.action).to(device)
    rewards = torch.FloatTensor(batch.reward).to(device)
    next_states = torch.FloatTensor(batch.next_state).to(device)
    dones = torch.FloatTensor(batch.done).to(device)
    
    # 计算当前Q值
    current_q = model(states)
    current_q = (current_q * actions).sum(dim=1)
    
    # 计算目标Q值
    with torch.no_grad():
        next_q = target_model(next_states)
        max_next_q = next_q.max(dim=1)[0]
        target_q = rewards + (1 - dones) * model.gamma * max_next_q
    
    # 计算损失
    loss = F.smooth_l1_loss(current_q, target_q)
    
    return loss

def train_model(env, model, config, logger):
    """训练模型"""
    try:
        # 初始化指标
        metrics = {
            'steps': 0,
            'total_reward': 0,
            'violations': 0,  # 添加违规计数
            'team_value': 0,
            'points': 0
        }
        
        # 训练循环
        for episode in range(config.num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_violations = 0  # 添加每轮违规计数
            
            while not done:
                # 选择动作
                action = model.select_action(state)
                
                # 执行动作
                # 新版本 Gym 返回 (next_state, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated  # 合并两种结束状态
                
                # 更新指标
                episode_reward += reward
                episode_steps += 1
                if 'violation' in info:
                    episode_violations += int(info['violation'])
                
                # 存储经验
                if hasattr(model, 'replay_buffer'):
                    model.replay_buffer.append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done
                    })
                
                # 更新状态
                state = next_state
                
                # 更新探索率
                if hasattr(model, 'update_epsilon'):
                    model.update_epsilon()
            
            # 更新总指标
            metrics['steps'] += episode_steps
            metrics['total_reward'] += episode_reward
            metrics['violations'] += episode_violations
            if 'team_value' in info:
                metrics['team_value'] = info['team_value']
            if 'points' in info:
                metrics['points'] = info['points']
            
            # 记录训练日志
            logger.log_training(episode, {
                'steps': metrics['steps'],
                'total_reward': metrics['total_reward'],
                'avg_reward': metrics['total_reward'] / (episode + 1),
                'epsilon': model.epsilon if hasattr(model, 'epsilon') else 0,
                'violations': metrics['violations'],
                'team_value': metrics['team_value'],
                'points': metrics['points']
            })
            
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise