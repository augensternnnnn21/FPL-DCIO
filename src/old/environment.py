import os
import numpy as np
import pandas as pd
import gymnasium as gym
from pathlib import Path
from config import PROCESSED_DATA_PATH, NUM_PLAYERS

class FPLEnv(gym.Env):
    def __init__(self, season="2016-17", verbose=True):
        super().__init__()
        self.verbose = verbose
        
        try:
            # 修正数据路径
            base_path = Path(__file__).parent.parent.parent / "data"  # 回到项目根目录的data文件夹
            
            # 尝试多个可能的路径
            possible_paths = [
                base_path / season / f"{season}_processed.csv",  # /data/2016-17/2016-17_processed.csv
                base_path / "processed" / f"{season}_processed.csv",  # /data/processed/2016-17_processed.csv
                base_path / "cleaned_merged_seasons.csv"  # /data/cleaned_merged_seasons.csv
            ]
            
            data_loaded = False
            for path in possible_paths:
                if path.exists():
                    if path.name == "cleaned_merged_seasons.csv":
                        all_data = pd.read_csv(path)
                        self.data = all_data[all_data['season'] == season].copy()
                    else:
                        self.data = pd.read_csv(path)
                    data_loaded = True
                    if self.verbose:
                        print(f"\nLoaded data from: {path}")
                    break
            
            if not data_loaded:
                raise FileNotFoundError(
                    f"Could not find data file in any of these locations:\n" + 
                    "\n".join(str(p) for p in possible_paths)
                )
            
            if self.verbose:
                print(f"Data shape: {self.data.shape}")
                print(f"Columns: {self.data.columns.tolist()}")
            
            # 初始化环境状态
            self.reset()
            
            # 定义动作空间和观察空间
            self.action_space = gym.spaces.Discrete(NUM_PLAYERS)
            
            # 观察空间：每个球员的特征
            num_features = len(self.get_state_features())
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(NUM_PLAYERS, num_features),
                dtype=np.float32
            )
            
        except Exception as e:
            print(f"Error initializing environment: {str(e)}")
            raise
    
    def get_state_features(self):
        """获取状态特征列表"""
        base_features = [
            'price', 'points', 'minutes', 'goals_scored',
            'assists', 'clean_sheets', 'goals_conceded'
        ]
        
        # 检查可选特征是否存在
        optional_features = ['bonus', 'bps']
        available_features = [f for f in optional_features if f in self.data.columns]
        
        return base_features + available_features
    
    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置状态
        self.current_gameweek = 1
        self.selected_players = []
        self.total_value = 0
        self.total_points = 0
        
        # 获取当前状态
        state = self._get_state()
        
        return state, {}  # 返回状态和空的info字典
    
    def _get_state(self):
        print("可用的列:", self.data.columns.tolist())
        print("数据预览:\n", self.data.head())
        """获取当前状态"""
        # 获取当前比赛周的数据
        current_data = self.data[self.data['GW'] == self.current_gameweek]
        
        # 提取特征
        features = self.get_state_features()
        state = current_data[features].values
        
        # 标准化
        state = (state - np.mean(state, axis=0)) / (np.std(state, axis=0) + 1e-8)
        
        return state
    
    def step(self, action):
        # 获取当前比赛周的数据
        gw_data = self.data[self.data['GW'] == self.current_gameweek].copy()
        
        # 获取当前比赛周的球员名单
        current_gw_players = set(gw_data['name'])
        
        valid_actions = []
        
        for idx in action:
            if idx < NUM_PLAYERS:
                player_name = gw_data[gw_data['name'] == idx]['name'].values[0]
                if player_name in current_gw_players:
                    valid_actions.append(idx)
        
        # 验证选择
        selected_players = gw_data[gw_data['name'].isin(valid_actions)]
        
        total_cost = selected_players['price'].sum()
        
        # 检查约束
        budget_ok = total_cost <= BUDGET
        
        if not budget_ok:
            reward = -min(total_cost - BUDGET, 100)
            if self.verbose:
                print(f"预算超支: {total_cost - BUDGET:.1f}，奖励: {reward:.1f}")
            return self._get_state(), reward, True, False, {}
        
        # 计算奖励和更新状态
        total_points = selected_players['points'].sum()
        self.total_points += total_points
        self.total_value += total_cost
        
        # 更新比赛周
        self.current_gameweek += 1
        done = self.current_gameweek > 38
        
        return self._get_state(), total_points, done, False, {}
