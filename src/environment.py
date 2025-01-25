import os
import gymnasium as gym
import numpy as np
import pickle
from gymnasium import spaces
from config import PROCESSED_DATA_PATH, BUDGET, NUM_PLAYERS
import pandas as pd
import torch

class FPLEnv(gym.Env):
    def __init__(self, season: str = "2022-23", verbose=False):
        super().__init__()
        self.verbose = verbose
        self.default_season = season
        self.current_season = season
        
        # 加载和预处理数据
        self._load_data()
        
        # 添加新的特征计算
        self._add_advanced_features()
        
        # 计算每个比赛周的最大积分
        self.max_points_by_gw = self.data.groupby('gw')['total_points'].max()
        self.cumulative_max_points = self.max_points_by_gw.cumsum()
        
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
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_PLAYERS, 12)),
            'position_onehot': spaces.Box(low=0, high=1, shape=(NUM_PLAYERS, 4)),
            'team_onehot': spaces.Box(low=0, high=1, shape=(NUM_PLAYERS, 20)),
            'values': spaces.Box(low=0, high=np.inf, shape=(NUM_PLAYERS,)),
            'history': spaces.Box(low=-np.inf, high=np.inf, shape=(5, NUM_PLAYERS, 12))
        })
        self.action_space = spaces.Discrete(NUM_PLAYERS)
        
        # 初始化状态
        self.current_gw = 1
        self.current_team = []
        self.current_budget = BUDGET
        
        # 添加阵容要求
        self.formation_requirements = {
            'GK': 2,
            'DEF': 5,
            'MID': 5,
            'FWD': 3
        }
        
        # 加载最高分数据
        self._load_max_points()
    
    def _load_data(self):
        """加载数据"""
        data_path = os.path.join(PROCESSED_DATA_PATH, f"{self.current_season}_processed.pkl")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No processed data found for season {self.current_season}")
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # 确保数据已排序
        self.data = self.data.sort_values(['name', 'gw']).reset_index(drop=True)
        
        # 添加高级特征
        self._add_advanced_features()
        
        # 数据验证
        if self.verbose:
            print("\nData validation:")
            print("Position distribution:")
            print(self.data['position'].value_counts(dropna=False))
            print("\nFeatures available:")
            print(self.data.columns.tolist())
        
        self.data = self.data.reset_index(drop=True)  # 确保索引从0开始连续
        self.player_id_map = {name: idx for idx, name in enumerate(self.data['name'].unique())}
        self.inverse_id_map = {v: k for k, v in self.player_id_map.items()}  # 新增反向映射
        
        self.num_players = len(self.data['name'].unique())
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_PLAYERS, 12)),
            'position_onehot': spaces.Box(low=0, high=1, shape=(NUM_PLAYERS, 4)),
            'team_onehot': spaces.Box(low=0, high=1, shape=(NUM_PLAYERS, 20)),
            'values': spaces.Box(low=0, high=np.inf, shape=(NUM_PLAYERS,)),
            'history': spaces.Box(low=-np.inf, high=np.inf, shape=(5, NUM_PLAYERS, 12))
        })
        self.action_space = spaces.Discrete(NUM_PLAYERS)
        
        # 初始化状态
        self.current_gw = 1
        self.current_team = []
        self.current_budget = BUDGET
        
        # 添加阵容要求
        self.formation_requirements = {
            'GK': 2,
            'DEF': 5,
            'MID': 5,
            'FWD': 3
        }
        
        # 加载最高分数据
        self._load_max_points()
    
    def _load_max_points(self):
        """加载每个比赛周的最高得分"""
        self.max_points_by_gw = pd.Series(index=range(1, 39))
        for gw in range(1, 39):
            gw_data = self.data[self.data['gw'] == gw]
            if not gw_data.empty:
                self.max_points_by_gw[gw] = gw_data['total_points'].max()
            else:
                self.max_points_by_gw[gw] = 0
    
    def _add_advanced_features(self):
        """添加高级特征"""
        try:
            # 1. 计算最近5轮的统计数据
            self.data['recent_points_mean'] = self.data.groupby('name')['total_points'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
            self.data['recent_points_std'] = self.data.groupby('name')['total_points'].transform(
                lambda x: x.rolling(5, min_periods=1).std().fillna(0)
            )
            
            # 2. 计算状态/表现
            self.data['form'] = self.data.groupby('name')['total_points'].transform(
                lambda x: x.ewm(span=5).mean()
            )
            
            # 3. 计算性价比
            self.data['price_value'] = self.data['total_points'] / self.data['value']
            
            # 4. 计算球队实力
            team_strength = self.data.groupby(['team', 'gw'])['total_points'].mean().reset_index()
            team_strength['team_strength'] = team_strength.groupby('team')['total_points'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
            self.data = self.data.merge(
                team_strength[['team', 'gw', 'team_strength']],
                on=['team', 'gw'],
                how='left'
            )
            
            # 5. 计算位置稀缺性
            pos_counts = self.data.groupby(['gw', 'position']).size().reset_index(name='count')
            pos_counts['position_scarcity'] = pos_counts.groupby('position')['count'].transform(
                lambda x: 1/x
            )
            self.data = self.data.merge(
                pos_counts[['gw', 'position', 'position_scarcity']],
                on=['gw', 'position'],
                how='left'
            )
            
            # 6. 添加主场/客场标记
            if 'is_home' not in self.data.columns:
                self.data['is_home'] = np.random.choice([0, 1], size=len(self.data))  # 临时解决方案
            
            # 7. 添加对手防守强度
            if 'opponent_defense_strength' not in self.data.columns:
                self.data['opponent_defense_strength'] = self.data.groupby('team')['team_strength'].transform(
                    lambda x: 1 - (x / x.max())  # 归一化的防守强度
                )
            
            # 8. 添加比赛难度
            if 'fixture_difficulty' not in self.data.columns:
                self.data['fixture_difficulty'] = (
                    self.data['opponent_defense_strength'] * 0.7 +
                    (1 - self.data['is_home']) * 0.3
                )
            
            # 填充缺失值
            self.data = self.data.fillna(0)
            
            if self.verbose:
                print("\nAdvanced features added successfully")
                print("Sample of processed data:")
                print(self.data[['name', 'gw', 'recent_points_mean', 'form', 'team_strength']].head())
            
        except Exception as e:
            print(f"Error in _add_advanced_features: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _calculate_fixture_difficulty(self):
        """计算未来赛程难度"""
        # 1. 计算球队的整体表现
        team_stats = self.data.groupby(['gw', 'team']).agg({
            'total_points': ['mean', 'std'],  # 球队得分情况
            'minutes': 'sum',                 # 出场时间
            'value': 'mean',                  # 平均价值
        }).reset_index()

        # 2. 计算球队强度
        team_stats['team_strength'] = (
            0.6 * team_stats[('total_points', 'mean')] +  # 得分能力
            0.4 * team_stats[('value', 'mean')]          # 整体价值
        )
        
        # 3. 使用滚动平均计算趋势
        team_stats['team_strength'] = team_stats.groupby('team')['team_strength'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        
        # 4. 标准化球队强度
        team_stats['team_strength'] = (
            (team_stats['team_strength'] - team_stats['team_strength'].mean()) /
            team_stats['team_strength'].std()
        )
        
        # 5. 合并到主数据
        strength_data = team_stats[['gw', 'team', 'team_strength']].copy()
        
        # 6. 计算比赛难度（直接使用球队强度的负值）
        self.data['fixture_difficulty'] = -self.data['team_strength']  # 负号表示对手强度越高难度越大
        
        # 7. 标准化难度
        self.data['fixture_difficulty'] = (
            (self.data['fixture_difficulty'] - self.data['fixture_difficulty'].mean()) /
            self.data['fixture_difficulty'].std()
        )
    
    def _calculate_formation_balance(self, selected_players):
        """计算阵容结构的平衡性"""
        positions = selected_players['position'].value_counts()
        
        # 1. 检查基本要求
        if positions.get('GK', 0) != 2:
            return -1.0
        if positions.get('DEF', 0) < 3 or positions.get('MID', 0) < 3 or positions.get('FWD', 0) < 1:
            return -1.0
        
        # 2. 计算平衡性得分
        balance_score = 0.0
        
        # 后场平衡（3-5后卫）
        def_count = positions.get('DEF', 0)
        if 3 <= def_count <= 5:
            balance_score += 0.3
        
        # 中场厚度（至少4名中场）
        mid_count = positions.get('MID', 0)
        if mid_count >= 4:
            balance_score += 0.4
        
        # 锋线合理性（1-3前锋）
        fwd_count = positions.get('FWD', 0)
        if 1 <= fwd_count <= 3:
            balance_score += 0.3
        
        return balance_score
    
    def _evaluate_fixture_potential(self, selected_players):
        """评估未来比赛的潜力"""
        # 1. 对手防守强度（越弱越好）
        defense_potential = 1 - selected_players['opponent_defense_strength'].mean()
        
        # 2. 主场优势
        home_advantage = selected_players['is_home'].mean()
        
        # 3. 球员状态
        form_potential = selected_players['form'].mean() / selected_players['form'].max()
        
        # 综合评分
        potential_score = (
            0.4 * defense_potential +
            0.3 * home_advantage +
            0.3 * form_potential
        )
        
        return potential_score
    
    def _get_state(self):
        """获取当前状态"""
        current_gw_data = self.data[self.data['gw'] == self.current_gw].copy()
        
        # 添加检查
        num_players = current_gw_data.shape[0]
        assert num_players > 1, f"当前比赛周球员数量不足：{num_players}"
        
        # 基础特征
        base_features = np.column_stack([
            current_gw_data['recent_points_mean'],
            current_gw_data['recent_points_std'],
            current_gw_data['form'],
            current_gw_data['fixture_difficulty'],
            current_gw_data['price_value'],
            current_gw_data['team_strength'],
            current_gw_data['position_scarcity'],
            current_gw_data['value'],
            current_gw_data['total_points'],
            current_gw_data['is_home'],
            current_gw_data['opponent_defense_strength'],
            current_gw_data['fixture_difficulty']
        ])
        
        # 位置的one-hot编码
        position_onehot = pd.get_dummies(
            current_gw_data['position'],
            columns=['GK', 'DEF', 'MID', 'FWD']
        ).values
        
        # 球队的one-hot编码
        team_onehot = pd.get_dummies(
            current_gw_data['team'],
            prefix='team'
        ).values
        
        # 获取历史数据
        history_features = []
        for gw in range(max(1, self.current_gw - 5), self.current_gw):
            gw_data = self.data[self.data['gw'] == gw]
            if not gw_data.empty:
                gw_features = np.column_stack([
                    gw_data['total_points'],
                    gw_data['value'],
                    gw_data['form'],
                    gw_data['team_strength'],
                    gw_data['fixture_difficulty'],
                    gw_data['is_home'],
                    gw_data['opponent_defense_strength'],
                    gw_data['position_scarcity'],
                    gw_data['recent_points_mean'],
                    gw_data['recent_points_std'],
                    gw_data['price_value'],
                    gw_data['fixture_difficulty']
                ])
            else:
                gw_features = np.zeros((len(current_gw_data), 12))
            history_features.append(gw_features)
        
        # 填充历史数据到5个时间步
        while len(history_features) < 5:
            history_features.insert(0, np.zeros((len(current_gw_data), 12)))
        
        # 转换为numpy数组
        history_features = np.stack(history_features)
        
        # 确保所有特征都是float32类型
        base_features = base_features.astype(np.float32)
        position_onehot = position_onehot.astype(np.float32)
        team_onehot = team_onehot.astype(np.float32)
        history_features = history_features.astype(np.float32)
        
        # 转换为PyTorch张量并移动到正确的设备
        return {
            'features': torch.FloatTensor(base_features),
            'position_onehot': torch.FloatTensor(position_onehot),
            'team_onehot': torch.FloatTensor(team_onehot),
            'values': torch.FloatTensor(current_gw_data['value'].values),
            'history': torch.FloatTensor(history_features)
        }
    
    def reset(self, seed=None, options=None, season=None):
        """
        重置环境
        Args:
            seed: 随机种子
            options: 其他选项
            season: 指定赛季，如果为None则使用默认赛季
        """
        super().reset(seed=seed)
        
        # 如果指定了新赛季，则重新加载数据
        if season is not None and season != self.current_season:
            self.current_season = season
            self._load_data()
            self._load_max_points()
        
        # 重置状态
        self.current_gw = 1
        self.current_team = []
        self.current_budget = BUDGET
        
        # 确保返回的状态是字典
        state = self._get_state()
        return state, {}
    
    def step(self, action):
        """执行动作"""
        # 获取当前比赛周的数据
        gw_data = self.data[self.data['gw'] == self.current_gw].copy()
        
        # 获取当前比赛周的球员名单
        current_gw_players = set(gw_data['name'])
        
        valid_actions = []
        
        for idx in action:
            if idx < self.num_players:
                player_name = self.inverse_id_map.get(idx)
                if player_name in current_gw_players:
                    valid_actions.append(idx)
        
        # 验证选择
        selected_players = self.data[
            (self.data['gw'] == self.current_gw) & 
            (self.data['name'].isin([self.inverse_id_map[idx] for idx in valid_actions]))
        ]
        
        total_cost = selected_players['value'].sum()
        
        # 检查约束
        positions = selected_players['position'].value_counts()
        budget_ok = total_cost <= self.budget
        
        # 初始化奖励
        reward = 0
        
        if not budget_ok:
            reward = -min(total_cost - self.budget, 100)
            if self.verbose:
                print(f"预算超支: {total_cost - self.budget:.1f}，奖励: {reward:.1f}")
            # 确保返回的状态是字典
            next_state = self._get_state()
            return next_state, reward, True, False, {}

        # 计算合规性奖励（位置约束满足度）
        position_reward = sum([1 if positions.get(pos, 0) == req else -0.2 for pos, req in self.formation_requirements.items()])
        budget_reward = 0.5 * (total_cost / BUDGET)  # 鼓励使用预算

        # 新增：对手强度惩罚（对手过去5场平均失球数越少，得分难度越高）
        team_strength = selected_players['team_strength'].mean()
        strength_bonus = 0.2 * team_strength  # 使用球队强度作为奖励组件
        
        # 计算得分奖励
        total_points = selected_players['total_points'].sum()
        
        # 处理第一周的特殊情况
        if self.current_gw == 1:
            current_max_points = self.data[self.data['gw'] == 1]['total_points'].max()
        else:
            current_max_points = self.data[self.data['gw'] == self.current_gw]['total_points'].max()
        
        # 添加阵容稳定性约束
        stability_penalty = 0
        if self.current_gw > 1:
            cost_ratio_diff = (total_cost / BUDGET) - self.last_cost_ratio
            stability_penalty = -0.1 * (cost_ratio_diff ** 2)
        self.last_cost_ratio = total_cost / BUDGET
        
        # 避免除以0
        current_max_points = max(current_max_points, 1)
        points_reward = total_points / current_max_points
        
        # 计算总奖励
        reward = (0.7 * points_reward + 
                 0.2 * position_reward + 
                 0.1 * budget_reward + 
                 strength_bonus + 
                 stability_penalty)
        
        if self.verbose:
            print(f"得分奖励: {points_reward:.3f}")
            print(f"合规性奖励: {position_reward:.3f}")
            print(f"预算奖励: {budget_reward:.3f}")
            print(f"总奖励: {reward:.3f}")
            print(f"本周得分: {total_points:.1f}")
            print(f"本周最高分: {self.max_points_by_gw.loc[self.current_gw]:.1f}")
        
        # 更新状态
        self.current_team = valid_actions
        self.current_budget = self.budget - total_cost
        
        # 更新比赛周
        self.current_gw += 1
        done = self.current_gw > 38
        
        # 确保返回的状态是字典
        next_state = self._get_state()
        return next_state, reward, done, False, {}
