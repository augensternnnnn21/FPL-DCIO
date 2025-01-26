import os
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
from config import ExperimentConfig

from data.data_loader import FPLDataLoader
from data.preprocessor import FPLPreprocessor
from models.differentiable_constraints import DifferentiableChecker
from models.st_conv import DilatedTCN


class FPLEnv(gym.Env):
    """FPL环境"""
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        
        # 直接使用具体的文件路径
        base_dir = Path("/Users/zhangyufei/Desktop/FPL-DCIO/data")
        self.seasons_file = base_dir / "cleaned_merged_seasons.csv"
        self.teams_file = base_dir / "master_team_list.csv"
        
        print(f"\n初始化环境")
        print(f"📄 赛季数据文件: {self.seasons_file}")
        print(f"📄 球队数据文件: {self.teams_file}")
        
        # 验证数据文件
        self._validate_data_files()
        
        try:
            # 初始化数据加载器并加载数据
            self.data_loader = FPLDataLoader(
                seasons_path=self.seasons_file,
                teams_path=self.teams_file
            )
            raw_data = self.data_loader.load_data()
            self.team_data = self.data_loader.load_team_list()
            
        except Exception as e:
            print(f"❌ 数据加载错误: {str(e)}")
            raise
            
        self.budget = config.budget
        self.max_players = config.max_players
        
        # 初始化组件
        self.preprocessor = FPLPreprocessor()
        self.constraint_checker = DifferentiableChecker(config.budget, config.max_players)
        
        # 加载和预处理数据
        try:
            self.data = self.preprocessor.preprocess(raw_data, self.team_data)
            self._validate_data()
            self.all_players = self.data['name'].unique()
            print(f"✅ 成功加载 {len(self.all_players)} 名球员数据")
        except Exception as e:
            print(f"❌ 数据预处理错误: {str(e)}")
            raise
        
        # 保存参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化空间
        self._init_spaces()
        
        # 重置环境
        self.reset()
    
    def _validate_data(self):
        """验证数据完整性"""
        required_columns = ['value', 'total_points', 'minutes', 'goals_scored',
                          'assists', 'clean_sheets', 'goals_conceded', 'position', 'team']
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
    
    def _init_spaces(self):
        """初始化动作和观察空间"""
        num_features = 12  # 基本特征数量
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.all_players), num_features),
            dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(len(self.all_players))
    
    def reset(self, season: Optional[str] = None) -> Dict:
        """重置环境"""
        if season:
            try:
                raw_data = self.data_loader.load_data(season)
                self.data = self.preprocessor.preprocess(raw_data, self.team_data)
            except Exception as e:
                print(f"❌ 加载赛季数据错误: {str(e)}")
                raise
        
        self.current_gameweek = 1
        self.selected_players = np.zeros(len(self.data[self.data['gw'] == 1]))
        self.available_budget = self.budget
        
        # 获取当前状态的数据
        current_data = self.data[self.data['gw'] == self.current_gameweek]
        
        # 返回初始状态
        return {
            'player_features': torch.from_numpy(self._get_state_features(current_data)).float().to(self.device),
            'position_onehot': torch.from_numpy(self._get_position_onehot(current_data)).float().to(self.device),
            'team_onehot': self._get_team_onehot(current_data).to(self.device),
            'gameweek': self.current_gameweek,
            'raw_features': {
                'values': torch.from_numpy(current_data['value'].values).float().to(self.device),
                'total_points': torch.from_numpy(current_data['total_points'].values).float().to(self.device),
                'names': current_data['name'].values
            }
        }
    
    def step(self, action):
        """执行一步动作"""
        # 验证动作
        if not isinstance(action, np.ndarray):
            action = action.cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)
        
        # 获取当前状态的数据
        current_data = self.data[self.data['gw'] == self.current_gameweek]
        
        # 确保 action 是正确的维度
        if len(action.shape) == 1:
            action = action.reshape(-1)  # 确保是一维数组
        
        # 更新选择状态
        self.selected_players = np.array(action, dtype=np.float32)  # 确保是 numpy 数组
        
        # 计算奖励
        reward = self._calculate_reward(action, current_data)
        
        # 更新游戏周
        self.current_gameweek += 1
        done = self.current_gameweek > 38
        
        # 检查约束违反
        violations = {
            'budget': self._check_budget_constraint(action, current_data),
            'squad_size': self._check_squad_size_constraint(action),
            'position': self._check_position_constraint(action, current_data)
        }
        
        # 如果违反约束，给予惩罚
        if any(violations.values()):
            reward -= sum(violations.values()) * 1000  # 可以调整惩罚系数
        
        # 获取新状态
        next_state = {
            'player_features': torch.from_numpy(self._get_state_features(current_data)).float().to(self.device),
            'position_onehot': torch.from_numpy(self._get_position_onehot(current_data)).float().to(self.device),
            'team_onehot': self._get_team_onehot(current_data).to(self.device),
            'gameweek': self.current_gameweek,
            'raw_features': {
                'values': torch.from_numpy(current_data['value'].values).float().to(self.device),
                'total_points': torch.from_numpy(current_data['total_points'].values).float().to(self.device),
                'names': current_data['name'].values
            }
        }
        
        return next_state, reward, done, violations

    def _check_budget_constraint(self, action, current_data):
        """检查预算约束"""
        total_value = np.sum(current_data['value'].values * action)
        return max(0, total_value - self.budget)

    def _check_squad_size_constraint(self, action):
        """检查球队规模约束"""
        num_selected = np.sum(action)
        return abs(num_selected - self.max_players)

    def _check_position_constraint(self, action, current_data):
        """检查位置约束"""
        position_counts = {
            'GK': np.sum(action * (current_data['position'] == 'GK')),
            'DEF': np.sum(action * (current_data['position'] == 'DEF')),
            'MID': np.sum(action * (current_data['position'] == 'MID')),
            'FWD': np.sum(action * (current_data['position'] == 'FWD'))
        }
        
        violations = 0
        if position_counts['GK'] != 2:  # 需要2名门将
            violations += abs(2 - position_counts['GK'])
        if position_counts['DEF'] < 3 or position_counts['DEF'] > 5:  # 3-5名后卫
            violations += 1
        if position_counts['MID'] < 3 or position_counts['MID'] > 5:  # 3-5名中场
            violations += 1
        if position_counts['FWD'] < 1 or position_counts['FWD'] > 3:  # 1-3名前锋
            violations += 1
        
        return violations

    def _get_state_features(self, current_data):
        """获取状态特征"""
        print("\nDEBUG: Getting state features...")
        print(f"Current data shape: {current_data.shape}")
        
        # 获取位置的独热编码
        position_onehot = self._get_position_onehot(current_data)
        print(f"Position onehot shape: {position_onehot.shape}")
        
        # 获取数值特征
        basic_features = ['value', 'total_points', 'minutes', 'goals_scored',
                         'assists', 'clean_sheets', 'goals_conceded']
        numerical_features = current_data[basic_features].values
        print(f"Numerical features shape: {numerical_features.shape}")
        
        # 初始化或验证 selected_players
        print(f"\nDEBUG: Selected players info:")
        print(f"Type: {type(self.selected_players)}")
        print(f"Value: {self.selected_players}")
        
        # 确保 selected_players 是正确维度的数组
        if isinstance(self.selected_players, (int, float, np.ndarray)):
            # 如果是标量或者标量数组，创建新数组
            if np.isscalar(self.selected_players) or (isinstance(self.selected_players, np.ndarray) and self.selected_players.shape == ()):
                print("Converting scalar to array")
                self.selected_players = np.full(len(current_data), float(self.selected_players), dtype=np.float32)
            elif len(self.selected_players) != len(current_data):
                print(f"Resizing array from {len(self.selected_players)} to {len(current_data)}")
                self.selected_players = np.zeros(len(current_data), dtype=np.float32)
        else:
            print("Creating new array")
            self.selected_players = np.zeros(len(current_data), dtype=np.float32)
        
        print(f"\nDEBUG: After processing:")
        print(f"Selected players type: {type(self.selected_players)}")
        print(f"Selected players shape: {self.selected_players.shape}")
        
        # 创建预算特征
        budget_feature = np.full((len(current_data), 1), self.available_budget / 100.0)
        print(f"\nDEBUG: Budget feature shape: {budget_feature.shape}")
        
        # 组合所有特征
        try:
            state = np.concatenate([
                position_onehot,
                numerical_features,
                self.selected_players.reshape(-1, 1),
                budget_feature
            ], axis=1)
            print(f"\nDEBUG: Final state shape: {state.shape}")
            return state
        except Exception as e:
            print(f"\nERROR: Failed to concatenate features:")
            print(f"Position onehot: {position_onehot.shape}")
            print(f"Numerical features: {numerical_features.shape}")
            print(f"Selected players: {self.selected_players.shape}")
            print(f"Budget feature: {budget_feature.shape}")
            print(f"Error: {str(e)}")
            raise
    
    def _get_position_onehot(self, data):
        """获取位置的独热编码"""
        # 首先创建位置映射
        position_mapping = {
            'Goalkeeper': 'GK',
            'Defender': 'DEF',
            'Midfielder': 'MID',
            'Forward': 'FWD',
            # 添加可能的其他映射
            'GK': 'GK',
            'DEF': 'DEF',
            'MID': 'MID',
            'FWD': 'FWD'
        }
        
        # 转换位置名称
        data['position_mapped'] = data['position'].map(position_mapping)
        
        # 检查是否有未映射的位置
        unmapped = data[data['position_mapped'].isna()]['position'].unique()
        if len(unmapped) > 0:
            print(f"Warning: Found unmapped positions: {unmapped}")
            # 可以添加默认映射或者抛出错误
        
        # 获取独热编码
        positions = ['GK', 'DEF', 'MID', 'FWD']
        position_onehot = pd.get_dummies(data['position_mapped'])[positions].values
        
        return position_onehot
    
    def _get_team_onehot(self, data: pd.DataFrame) -> torch.Tensor:
        """获取球队的独热编码"""
        team_onehot = pd.get_dummies(data['team']).values
        return torch.from_numpy(team_onehot).float()

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

    def _load_team_data(self) -> pd.DataFrame:
        """加载球队数据"""
        try:
            team_data = pd.read_csv(self.teams_file)
            print(f"✅ 成功加载 {len(team_data)} 支球队数据")
            return team_data
        except Exception as e:
            print(f"❌ 球队数据加载失败: {str(e)}")
            raise

    def _validate_data_files(self):
        """验证数据文件"""
        if not self.seasons_file.exists():
            raise FileNotFoundError(f"找不到赛季数据文件: {self.seasons_file}")
            
        if not self.teams_file.exists():
            raise FileNotFoundError(f"找不到球队数据文件: {self.teams_file}")
            
        if self.seasons_file.stat().st_size == 0:
            raise ValueError(f"赛季数据文件为空: {self.seasons_file}")
            
        if self.teams_file.stat().st_size == 0:
            raise ValueError(f"球队数据文件为空: {self.teams_file}")
            
        print("✅ 数据文件验证通过")

    def _calculate_reward(self, action, current_data):
        """计算奖励值
        
        Args:
            action: 选择的球员 (numpy array)
            current_data: 当前比赛周的数据 (DataFrame)
        
        Returns:
            float: 奖励值
        """
        try:
            # 获取选中球员的总分
            selected_points = current_data['total_points'].values * action
            total_points = np.sum(selected_points)
            
            # 获取选中球员的总价值
            selected_values = current_data['value'].values * action
            total_value = np.sum(selected_values)
            
            # 计算基础奖励（基于得分）
            reward = total_points
            
            # 添加预算约束的惩罚
            if total_value > self.budget:
                reward -= (total_value - self.budget) * 10  # 可以调整惩罚系数
            
            # 添加球员数量约束的惩罚
            num_selected = np.sum(action)
            if num_selected > self.max_players:
                reward -= (num_selected - self.max_players) * 100  # 可以调整惩罚系数
            elif num_selected < self.max_players:
                reward -= (self.max_players - num_selected) * 100  # 可以调整惩罚系数
            
            # 添加位置约束的惩罚
            position_counts = {
                'GK': np.sum(action * (current_data['position'] == 'GK')),
                'DEF': np.sum(action * (current_data['position'] == 'DEF')),
                'MID': np.sum(action * (current_data['position'] == 'MID')),
                'FWD': np.sum(action * (current_data['position'] == 'FWD'))
            }
            
            # 检查每个位置的球员数量是否符合要求
            if position_counts['GK'] != 2:  # 需要2名门将
                reward -= abs(2 - position_counts['GK']) * 100
            if position_counts['DEF'] < 3 or position_counts['DEF'] > 5:  # 3-5名后卫
                reward -= 100
            if position_counts['MID'] < 3 or position_counts['MID'] > 5:  # 3-5名中场
                reward -= 100
            if position_counts['FWD'] < 1 or position_counts['FWD'] > 3:  # 1-3名前锋
                reward -= 100
            
            # 更新可用预算
            self.available_budget = self.budget - total_value
            
            return reward
            
        except Exception as e:
            print(f"Error calculating reward: {str(e)}")
            print(f"Action shape: {action.shape}")
            print(f"Current data shape: {current_data.shape}")
            print(f"Available columns: {current_data.columns}")
            raise
