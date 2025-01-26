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


class FPLEnv(gym.Env):
    """FPL环境"""
    def __init__(self, config):
        super().__init__()
        
        # 直接使用具体的文件路径
        self.base_dir = Path(config.data_path)
        # 修改这里：直接使用基础目录下的文件
        self.seasons_file = self.base_dir / "cleaned_merged_seasons.csv"
        self.teams_file = self.base_dir / "master_team_list.csv"
        
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
        
        # 特殊卡片标记
        self.special_cards = {
            'triple_captain': False,
            'bench_boost': False
        }
    
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
        try:
            if season:
                raw_data = self.data_loader.load_data(season)
                if len(raw_data) == 0:
                    print(f"⚠️ 警告: {season} 赛季数据为空，使用默认数据")
                    raw_data = self.data_loader.load_data()
                self.data = self.preprocessor.preprocess(raw_data, self.team_data)
                
            self.current_gameweek = 1
            current_data = self.data[self.data['gw'] == self.current_gameweek]
            
            # 验证数据
            if len(current_data) == 0:
                raise ValueError(f"当前比赛周 {self.current_gameweek} 没有数据")
            
            self.selected_players = np.zeros(len(current_data))
            self.available_budget = self.budget
            
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
        except Exception as e:
            print(f"❌ 重置环境错误: {str(e)}")
            print(f"Season: {season}")
            print(f"Data shape: {self.data.shape if hasattr(self, 'data') else 'No data'}")
            raise
    
    def step(self, action):
        """执行一步，改为累加得分的方式"""
        try:
            current_data = self.data[self.data['gw'] == self.current_gameweek]
            # Debug信息
            print(f"Debug - Environment step:")
            print(f"Action shape: {action.shape}")
            print(f"Current data shape: {len(current_data)}")
            
            # 确保动作维度与当前数据匹配
            if len(action) != len(current_data):
                print(f"Warning: Action shape {action.shape} does not match data shape {len(current_data)}")
                # 如果动作维度过大，截断到正确大小
                if len(action) > len(current_data):
                    action = action[:len(current_data)]
                # 如果动作维度过小，填充到正确大小
                else:
                    action = np.pad(action, (0, len(current_data) - len(action)))
            
            # (1) 计算基础reward（累加玩家得分）
            base_reward, info = self._calculate_reward(action, current_data)
            
            # (2) 如果使用三倍队长卡 -> 仅对队长球员进行三倍
            if self.special_cards['triple_captain']:
                # 假设队长是action数组中标记为 captain_index
                captain_index = self._designate_captain(action, current_data)
                captain_points = current_data['total_points'].values[captain_index]
                triple_bonus = 2 * captain_points  # 原本1倍 + 额外2倍
                base_reward += triple_bonus
                info['triple_captain_bonus'] = triple_bonus
            
            # (3) 如果使用Bench Boost卡 -> 替补也算分(假设替补位是action后几位)
            if self.special_cards['bench_boost']:
                bench_indices = self._get_bench_indices(action, current_data)
                bench_points = np.sum(current_data['total_points'].values[bench_indices])
                base_reward += bench_points
                info['bench_boost_bonus'] = bench_points
            
            # 更新状态
            self.current_gameweek += 1
            done = self.current_gameweek > 38
            
            # 获取下一个状态
            next_state = None if done else self._get_state()
            
            return next_state, base_reward, done, info
            
        except Exception as e:
            print(f"Error in step: {str(e)}")
            print(f"Action type: {type(action)}")
            print(f"Action shape: {action.shape}")
            print(f"Action value: {action}")
            raise

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
        try:
            # 获取位置的独热编码
            position_onehot = self._get_position_onehot(current_data)
            
            # 获取数值特征
            basic_features = ['value', 'total_points', 'minutes', 'goals_scored',
                             'assists', 'clean_sheets', 'goals_conceded']
            numerical_features = current_data[basic_features].values
            
            # 确保 selected_players 是正确维度的数组
            if len(self.selected_players) != len(current_data):
                self.selected_players = np.zeros(len(current_data), dtype=np.float32)
            
            # 创建预算特征
            budget_feature = np.full((len(current_data), 1), self.available_budget)
            
            # 组合所有特征
            state = np.concatenate([
                position_onehot,
                numerical_features,
                self.selected_players.reshape(-1, 1),
                budget_feature
            ], axis=1)
            
            return state
        except Exception as e:
            print(f"Error in _get_state_features: {str(e)}")
            raise
    
    def _get_position_onehot(self, data):
        """获取位置的独热编码"""
        try:
            # 首先创建位置映射
            position_mapping = {
                'Goalkeeper': 'GK',
                'Defender': 'DEF',
                'Midfielder': 'MID',
                'Forward': 'FWD',
                'GKP': 'GK',  # 添加这个映射
                # 添加可能的其他映射
                'GK': 'GK',
                'DEF': 'DEF',
                'MID': 'MID',
                'FWD': 'FWD'
            }
            
            # 转换位置名称
            data = data.copy()
            data['position_mapped'] = data['position'].map(position_mapping)
            
            # 检查是否有未映射的位置
            unmapped = data[data['position_mapped'].isna()]['position'].unique()
            if len(unmapped) > 0:
                print(f"Warning: Found unmapped positions: {unmapped}")
                # 可以添加默认映射
                data.loc[data['position_mapped'].isna(), 'position_mapped'] = 'GK'
            
            # 获取独热编码
            positions = ['GK', 'DEF', 'MID', 'FWD']
            position_onehot = pd.get_dummies(data['position_mapped'])
            
            # 确保所有需要的列都存在
            for pos in positions:
                if pos not in position_onehot.columns:
                    position_onehot[pos] = 0
            
            return position_onehot[positions].values
            
        except Exception as e:
            print(f"Error in _get_position_onehot: {str(e)}")
            print(f"Data shape: {data.shape}")
            print(f"Available columns: {data.columns.tolist()}")
            print(f"Position values: {data['position'].unique()}")
            raise
    
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
        """计算奖励"""
        try:
            # 检查硬性规则
            position_counts = {
                'GK': np.sum(action * (current_data['position'] == 'GK')),
                'DEF': np.sum(action * (current_data['position'] == 'DEF')),
                'MID': np.sum(action * (current_data['position'] == 'MID')),
                'FWD': np.sum(action * (current_data['position'] == 'FWD'))
            }
            
            total_cost = np.sum(current_data['value'].values * action)
            total_players = sum(position_counts.values())
            
            # 硬性规则检查
            if (position_counts['GK'] != 2 or 
                position_counts['DEF'] != 5 or 
                position_counts['MID'] != 5 or 
                position_counts['FWD'] != 3 or 
                total_players != 15 or 
                total_cost > self.budget):
                return -100, {'violations': True}
            
            # 新增：检查同队人数限制
            team_counts = {}
            for i, row in current_data.iterrows():
                if action[i] == 1:
                    t = row['team']
                    team_counts[t] = team_counts.get(t, 0) + 1
            # 如果任意球队超过3人，直接判负
            if any(count > 3 for count in team_counts.values()):
                return -100, {'violations': True}
            
            # 计算奖励
            reward = 0
            
            # 1. 球员得分奖励
            points = np.sum(current_data['total_points'].values * action)
            reward += points
            
            # 2. 球队多样性奖励
            teams = current_data['team'].values[action == 1]
            unique_teams = len(np.unique(teams))
            reward += 0.1 * unique_teams
            
            return reward, {
                'points': points,
                'unique_teams': unique_teams,
                'total_cost': total_cost,
                'violations': False
            }
            
        except Exception as e:
            print(f"Error calculating reward: {str(e)}")
            raise

    def _get_state(self):
        """获取当前状态"""
        try:
            current_data = self.data[self.data['gw'] == self.current_gameweek]
            
            # 基础特征
            state = {
                'player_features': self._get_state_features(current_data),
                'position_onehot': self._get_position_onehot(current_data),
                'team_onehot': self._get_team_onehot(current_data),
                'gameweek': self.current_gameweek,
                'raw_features': {
                    'values': current_data['value'].values,
                    'total_points': current_data['total_points'].values,
                    'names': current_data['name'].values,
                    'teams': current_data['team'].values,
                    'positions': current_data['position'].values
                }
            }
            
            # 如果启用协同机制,添加协同矩阵
            if hasattr(self, 'use_synergy') and self.use_synergy:
                synergy_matrix = self._build_synergy_matrix(current_data)
                state['raw_features']['synergy_matrix'] = synergy_matrix
            
            return state
            
        except Exception as e:
            print(f"Error in _get_state: {str(e)}")
            raise

    def _build_synergy_matrix(self, current_data):
        """构建球员协同矩阵"""
        n = len(current_data)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 1. 同队加成
                    if current_data.iloc[i]['team'] == current_data.iloc[j]['team']:
                        matrix[i,j] += 0.3
                    
                    # 2. 互补位置加成
                    pos_i = current_data.iloc[i]['position']
                    pos_j = current_data.iloc[j]['position']
                    if (pos_i == 'DEF' and pos_j == 'GK') or (pos_i == 'GK' and pos_j == 'DEF'):
                        matrix[i,j] += 0.2
                    elif (pos_i == 'MID' and pos_j in ['DEF','FWD']) or (pos_j == 'MID' and pos_i in ['DEF','FWD']):
                        matrix[i,j] += 0.15
                    
                    # 3. 历史配合(可选)
                    # matrix[i,j] += self._get_historical_synergy(player_i, player_j)
        
        return matrix

    def _designate_captain(self, action, current_data):
        """示例：简单地选出第一个非0的位置作为队长"""
        selected_indices = np.where(action == 1)[0]
        if len(selected_indices) > 0:
            return selected_indices[0]  # 这里可以替换更复杂的逻辑
        return None
    
    def _get_bench_indices(self, action, current_data):
        """示例：假设后4位是替补"""
        selected_indices = np.where(action == 1)[0]
        if len(selected_indices) > 11:  # 先保证首发11人
            return selected_indices[-4:]
        return []