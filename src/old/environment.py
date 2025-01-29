import os
import gymnasium as gym
import numpy as np
import pickle
from gymnasium import spaces
from config import PROCESSED_DATA_PATH, BUDGET, NUM_PLAYERS
import pandas as pd

class FPLEnv(gym.Env):
    def __init__(self, season: str, verbose=False):
        super(FPLEnv, self).__init__()
        self.verbose = verbose  # 添加调试标志
        
        # 加载预处理数据
        with open(os.path.join(PROCESSED_DATA_PATH, f"{season}_processed.pkl"), 'rb') as f:
            self.data = pickle.load(f)
        
        # 数据验证
        print("\nData validation:")
        print("Position distribution:")
        print(self.data['position'].value_counts(dropna=False))  # 包括 NaN 值
        print("\nSample of players with missing positions:")
        print(self.data[pd.isna(self.data['position'])][['name', 'team', 'value', 'total_points']].head())
        
        self.data = self.data.reset_index(drop=True)  # 确保索引从0开始连续
        self.player_id_map = {name: idx for idx, name in enumerate(self.data['name'].unique())}
        self.inverse_id_map = {v: k for k, v in self.player_id_map.items()}  # 新增反向映射
        
        self.num_players = len(self.data['name'].unique())
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_players, 7), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(self.num_players)
        
        # 初始化状态
        self.current_gw = 1
        self.budget = BUDGET  # 保持总预算的引用
        self.current_budget = self.budget  # 跟踪当前剩余预算
        self.current_team = []  # 跟踪当前已选球员
        
        # 添加阵容要求
        self.formation_requirements = {
            'GK': 2,
            'DEF': 5,
            'MID': 5,
            'FWD': 3
        }
    
    def _get_state(self):
        """生成当前比赛周的状态"""
        gw_data = self.data[self.data['gw'] == self.current_gw]
        state = np.zeros((NUM_PLAYERS, 5))
        
        # 标准化数值
        max_points = self.data['total_points'].max()
        max_value = self.data['value'].max()
        max_minutes = self.data['minutes'].max()
        
        for _, row in gw_data.iterrows():
            player_id = self.player_id_map[row['name']]
            state[player_id] = [
                row['total_points'] / max_points,
                row['value'] / max_value,
                row['minutes'] / max_minutes,
                1 if player_id in self.current_team else 0,  # 使用当前队伍信息
                self._encode_position(row['position'])
            ]
        return state
    
    def _encode_position(self, pos: str) -> float:
        """将球员位置编码为数值"""
        if pd.isna(pos):  # 检查是否为 nan
            print(f"Warning: Found player with unknown position")
            return 0.0  # 或者其他默认值
        return {'GK': 0.0, 'DEF': 0.25, 'MID': 0.5, 'FWD': 0.75}[pos]
    
    def reset(self):
        self.current_gw = 1
        self.current_budget = self.budget  # 重置预算
        self.current_team = []  # 重置球队
        return self._get_state(), {}
    
    def step(self, action):
        # 去重并确保索引有效
        unique_actions = list(set(action))
        valid_actions = [i for i in unique_actions if i < self.num_players]
        
        if len(valid_actions) != 15:
            if self.verbose:
                print(f"\n选择人数不合法: {len(valid_actions)}")
            return self._get_state(), -100, True, False, {}
        
        # 验证选择
        selected_players = self.data.iloc[valid_actions]
        total_cost = selected_players['value'].sum()
        
        # 检查约束
        positions = selected_players['position'].value_counts()
        formation_valid = all(
            positions.get(pos, 0) == req
            for pos, req in self.formation_requirements.items()
        )
        budget_ok = total_cost <= self.budget
        
        # 计算奖励和更新状态
        total_points = selected_players['total_points'].sum()
        max_season_points = self.data['total_points'].max()
        reward = 0
        
        # 只在调试模式下打印
        if self.verbose:
            print("\n选择结果:")
            print(f"位置分布: {dict(positions)}")
            print(f"总成本: {total_cost:.1f}")
        
        if not budget_ok:
            reward = -min(total_cost - self.budget, 100)
            if self.verbose:
                print(f"预算超支: {total_cost - self.budget:.1f}，奖励: {reward:.1f}")
            return self._get_state(), reward, True, False, {}
        
        if not formation_valid:
            reward = -50
            if self.verbose:
                print(f"阵容不合法，奖励: {reward}")
            return self._get_state(), reward, True, False, {}
        
        # 计算正向奖励
        points_reward = (total_points / max_season_points) * 100
        bonus_reward = 10
        reward = points_reward + bonus_reward
        
        if self.verbose:
            print(f"得分奖励: {points_reward:.1f}")
            print(f"额外奖励: {bonus_reward}")
            print(f"总奖励: {reward:.1f}")
        
        # 更新状态
        self.current_team = valid_actions
        self.current_budget = self.budget - total_cost
        
        # 更新比赛周
        self.current_gw += 1
        done = self.current_gw > 38
        
        return self._get_state(), reward, done, False, {}