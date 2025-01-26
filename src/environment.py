import os
import gymnasium as gym
import numpy as np
import pickle
from gymnasium import spaces
from config import PROCESSED_DATA_PATH, BUDGET, NUM_PLAYERS
import pandas as pd
import torch
from collections import defaultdict
from typing import Tuple, Dict, Any
from preprocess import preprocess_data

class FPLEnv(gym.Env):
    def __init__(self, data_path: str, budget: float = 100.0, max_players: int = 15, history_length: int = 5):
        """
        初始化FPL环境
        Args:
            data_path: 数据路径
            budget: 预算限制
            max_players: 最大球员数量
            history_length: 历史数据长度
        """
        super().__init__()
        
        # 保存参数
        self.data_path = data_path
        self.budget = budget
        self.max_players = max_players
        self.history_length = history_length
        
        # 加载和预处理数据
        try:
            self.data = preprocess_data(self.data_path)
            self.all_players = self.data['name'].unique()
            print(f"Total unique players: {len(self.all_players)}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        
        # 初始化状态空间
        self._init_spaces()
        
        # 重置环境
        self.reset()
    
    def _init_spaces(self):
        """初始化动作和观察空间"""
        # 状态空间：每个球员的特征向量
        num_features = 12  # 根据实际特征数量调整
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.all_players), num_features),
            dtype=np.float32
        )
        
        # 动作空间：选择或不选择每个球员
        self.action_space = spaces.MultiBinary(len(self.all_players))
    
    def reset(self, season=None):
        """重置环境状态
        
        Args:
            season: 可选的赛季参数，如果提供则使用指定赛季的数据
        
        Returns:
            初始状态
        """
        if season is not None:
            self.current_season = season
            # 重新加载数据
            self._load_data()
        
        # 重置其他状态
        self.current_gameweek = 1
        self.selected_players = []
        self.available_budget = self.budget
        self.history = []
        
        # 获取初始状态
        return self._get_state()
    
    def step(self, action):
        """执行一步动作"""
        # 获取当前比赛周的数据
        current_players = self._get_available_players()
        
        # 如果action是张量，转换为numpy数组
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # 确保action的长度与当前可用球员数量相同
        if len(action) > len(current_players):
            action = action[:len(current_players)]
        
        # 获取选中的球员
        selected_indices = np.where(action)[0]
        selected_players = current_players.iloc[selected_indices]
        
        # 计算实际奖励（使用总分）
        total_points = selected_players['total_points'].sum()
        reward = float(total_points) if isinstance(total_points, (int, float)) else 0.0
        
        # 添加约束惩罚
        position_counts = selected_players['position'].value_counts()
        team_counts = selected_players['team'].value_counts()
        
        # 1. 位置约束
        required_counts = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        for pos, required in required_counts.items():
            actual = position_counts.get(pos, 0)
            if actual != required:
                reward -= 100 * abs(actual - required)  # 增加惩罚力度
        
        # 2. 球队约束
        if team_counts.max() > 3:
            reward -= 100 * (team_counts.max() - 3)  # 增加惩罚力度
        
        # 3. 总人数约束
        if len(selected_players) != 15:
            reward -= 100 * abs(len(selected_players) - 15)  # 增加惩罚力度
        
        # 更新比赛周
        self.current_gameweek += 1
        done = self.current_gameweek > 38
        
        # 准备下一个状态
        next_state = self._get_state() if not done else None
        
        # 打印调试信息
        print(f"\nStep Info:")
        print(f"Gameweek: {self.current_gameweek}")
        print(f"Available players: {len(current_players)}")
        print(f"Selected players: {len(selected_players)}")
        print(f"Total points: {reward}")
        print(f"Position distribution: {dict(position_counts)}")
        print(f"Team distribution: {dict(team_counts)}")
        
        return next_state, reward, done, {
            'selected_players': selected_players,
            'total_points': reward,
            'position_counts': position_counts,
            'team_counts': team_counts,
            'available_players': len(current_players),
            'selected_count': len(selected_players)
        }

    def _add_team_info(self):
        """添加球队信息"""
        try:
            print("\nAdding team information...")
            if self.data is None:
                raise ValueError("Data not loaded")
                
            if 'team' in self.data.columns:
                print("Team information already exists")
                return
                
            print("Warning: Missing team columns, using numeric team IDs")
            # 如果需要，这里可以添加更多的队伍信息处理逻辑
            
        except Exception as e:
            print(f"Error adding team information: {e}")
            print(f"Available columns: {self.data.columns.tolist() if self.data is not None else 'No data loaded'}")

    def _create_player_mappings(self):
        """创建球员ID映射"""
        # 获取所有唯一的球员名字
        unique_players = self.data['name'].unique()
        
        # 创建正向和反向映射
        self.id_map = {name: idx for idx, name in enumerate(unique_players)}
        self.inverse_id_map = {idx: name for name, idx in self.id_map.items()}
        
        # 记录球员总数
        self.num_players = len(unique_players)
        print(f"\nTotal unique players: {self.num_players}")

    def _add_position_info(self):
        """添加位置信息"""
        try:
            # 根据element_type映射位置
            position_map = {
                1: 'GK',  # 门将
                2: 'DEF', # 后卫
                3: 'MID', # 中场
                4: 'FWD'  # 前锋
            }
            
            # 确保element_type列存在
            if 'element_type' not in self.data.columns:
                print("Warning: element_type column not found")
                return False
            
            # 映射位置
            self.data['position'] = self.data['element_type'].map(position_map)
            
            # 验证映射结果
            position_counts = self.data['position'].value_counts()
            print("\nPosition distribution:")
            print(position_counts)
            
            return True
            
        except Exception as e:
            print(f"Error adding position information: {e}")
            return False
