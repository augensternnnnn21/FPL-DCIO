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
        self.selected_players = []
        self.available_budget = self.budget
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """执行一步"""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # 获取当前比赛周的数据
        current_data = self.data[self.data['gw'] == self.current_gameweek]
        
        # 获取选中的球员
        selected_indices = np.where(action)[0]
        
        # 构建状态字典
        state = {
            'selections': torch.from_numpy(action).float().to(self.device),
            'values': torch.from_numpy(current_data['value'].values).float().to(self.device),
            'position_onehot': self._get_position_onehot(current_data).to(self.device),
            'team_onehot': self._get_team_onehot(current_data).to(self.device)
        }
        
        # 检查约束
        violations = self.constraint_checker(state)
        reward = -violations['total'].item()
        
        # 更新状态
        self.current_gameweek += 1
        done = self.current_gameweek > 38
        
        # 记录日志
        if self.current_gameweek % 10 == 0:
            print(f"\nGameweek: {self.current_gameweek}")
            print(f"Selected players: {len(selected_indices)}")
            print(f"Total value: {current_data.iloc[selected_indices]['value'].sum():.2f}")
            print(f"Reward: {reward:.2f}")
        
        return self._get_state(), reward, done, violations
    
    def _get_state(self) -> Dict:
        """获取当前状态"""
        current_data = self.data[self.data['gw'] == self.current_gameweek]
        
        # 构建球员特征矩阵
        feature_columns = [
            'value', 'total_points', 'minutes', 'goals_scored',
            'assists', 'clean_sheets', 'goals_conceded'
        ]
        player_features = current_data[feature_columns].values
        
        # 获取位置和球队的独热编码
        position_onehot = self._get_position_onehot(current_data)
        team_onehot = self._get_team_onehot(current_data)
        
        # 合并所有特征
        all_features = np.concatenate([
            player_features,
            position_onehot.numpy(),
            team_onehot.numpy()
        ], axis=1)
        
        return {
            'player_features': torch.from_numpy(all_features).float().to(self.device),
            'position_onehot': position_onehot.to(self.device),
            'team_onehot': team_onehot.to(self.device),
            'gameweek': self.current_gameweek,
            'raw_features': {
                'values': torch.from_numpy(current_data['value'].values).float().to(self.device),
                'total_points': torch.from_numpy(current_data['total_points'].values).float().to(self.device),
                'names': current_data['name'].values
            }
        }
    
    def _get_position_onehot(self, data: pd.DataFrame) -> torch.Tensor:
        """获取位置的独热编码"""
        positions = ['GK', 'DEF', 'MID', 'FWD']
        position_onehot = pd.get_dummies(data['position'])[positions].values
        return torch.from_numpy(position_onehot).float()
    
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
