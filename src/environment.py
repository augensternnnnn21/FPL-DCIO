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
        
        # 加载数据
        print(f"\nLoading data from {self.data_path}")
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Successfully loaded data with {len(self.data)} rows")
            
            # 验证数据
            required_columns = ['name', 'value', 'total_points', 'position', 'team', 'gw']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # 数据预处理
            self._preprocess_data()
            
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            print(f"Error: Data file is empty at {self.data_path}")
            raise
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        
        # 初始化状态空间
        self._init_spaces()
        
        # 重置环境
        self.reset()
    
    def _preprocess_data(self):
        """数据预处理"""
        # 确保数值列的类型正确
        numeric_columns = ['value', 'total_points']
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # 处理缺失值
        self.data = self.data.fillna(0)
        
        # 确保gameweek是整数
        self.data['gw'] = self.data['gw'].astype(int)
        
        # 获取所有唯一的球员
        self.all_players = self.data['name'].unique()
        print(f"Total unique players: {len(self.all_players)}")
        
        # 创建球员ID映射
        self.player_id_map = {name: idx for idx, name in enumerate(self.all_players)}
        self.inverse_id_map = {idx: name for name, idx in self.player_id_map.items()}
        
        # 添加球队和位置信息
        self._add_team_info()
        self._add_position_info()
    
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
    
    def _load_data(self):
        """加载数据"""
        try:
            # 如果指定了赛季，尝试加载该赛季的数据
            if self.current_season:
                # 首先尝试加载处理好的数据
                if os.path.exists(self.data_path):
                    print(f"Loading processed data from {self.data_path}")
                    data = pd.read_csv(self.data_path)
                    
                    # 重命名 gameweek 列为 gw
                    if 'gameweek' in data.columns:
                        data = data.rename(columns={'gameweek': 'gw'})
                    
                    return data
                
                # 如果没有处理好的数据，尝试从原始数据加载
                raw_data_path = os.path.join(
                    os.path.dirname(self.data_path),
                    'gws'
                )
                
                if os.path.exists(raw_data_path):
                    print(f"\nProcessing {self.current_season}...")
                    print(f"Loading data for season {raw_data_path}...")
                    
                    # 加载所有比赛周数据
                    all_gws = []
                    for i in range(1, 39):  # 假设有38个比赛周
                        gw_path = os.path.join(raw_data_path, f'gw{i}.csv')
                        if os.path.exists(gw_path):
                            try:
                                # 尝试不同的编码
                                try:
                                    gw_data = pd.read_csv(gw_path)
                                except UnicodeDecodeError:
                                    gw_data = pd.read_csv(gw_path, encoding='latin1')
                                
                                # 添加比赛周信息
                                gw_data['gw'] = i
                                
                                all_gws.append(gw_data)
                                print(f"Successfully loaded gameweek {i} with latin1 encoding")
                            except Exception as e:
                                print(f"Error loading gameweek {i}: {e}")
                                continue
                    
                    if all_gws:
                        data = pd.concat(all_gws, ignore_index=True)
                        print(f"Total gameweeks loaded: {len(all_gws)}")
                        
                        # 保存处理后的数据
                        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
                        data.to_csv(self.data_path, index=False)
                        print(f"Saved processed data to {self.data_path}")
                        return data
                    else:
                        print("No gameweek data found")
                        return None
                
                print(f"Data path not found: {self.data_path}")
                return None
            
            return None
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _infer_position(self, player_data):
        """
        根据球员统计数据推断位置
        """
        try:
            # 更复杂的启发式规则
            if player_data['saves'] > 0:
                return 'GKP'
            elif player_data['clearances_blocks_interceptions'] > 2:  # 增加阈值
                return 'DEF'
            elif player_data['goals_scored'] > 0 and player_data['assists'] == 0:
                return 'FWD'
            elif player_data['assists'] > 0:
                return 'MID'
            else:
                # 使用更多的统计数据
                if player_data['tackles'] > 2 or player_data['clearances_blocks_interceptions'] > 0:
                    return 'DEF'
                elif player_data['attempted_passes'] > 20:
                    return 'MID'
                else:
                    return 'FWD'
        except Exception as e:
            print(f"Error inferring position: {str(e)}")
            print("Available data:", {k: v for k, v in player_data.items() if k in ['saves', 'clearances_blocks_interceptions', 'goals_scored', 'assists', 'tackles', 'attempted_passes']})
            return 'MID'  # 默认为中场

    def _get_available_players(self):
        """获取当前比赛周可用的球员"""
        try:
            print(f"\nDEBUG: Getting players for gameweek {self.current_gameweek + 1}")
            print(f"Available columns: {self.data.columns.tolist()}")
            
            # 确保使用正确的列名
            gw_column = 'gw' if 'gw' in self.data.columns else 'round'
            current_gw_data = self.data[self.data[gw_column] == self.current_gameweek + 1].copy()
            
            print(f"Number of players found: {len(current_gw_data)}")
            if current_gw_data.empty:
                print(f"Warning: No players found for gameweek {self.current_gameweek + 1}")
            return current_gw_data
        except Exception as e:
            print(f"Error getting available players: {str(e)}")
            print(f"Current gameweek: {self.current_gameweek}")
            print(f"Data shape: {self.data.shape}")
            print(f"Data columns: {self.data.columns.tolist()}")
            raise
    
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
    
    def _get_position_onehot(self, players_data: pd.DataFrame) -> np.ndarray:
        """
        获取球员位置的独热编码
        """
        try:
            print("\nCreating position one-hot encoding...")
            print(f"Data shape: {players_data.shape}")
            print(f"Position distribution:\n{players_data['position'].value_counts()}")
            
            # 创建独热编码矩阵
            position_onehot = np.zeros((len(players_data), self.num_positions))
            
            # 为每个球员设置位置
            for i, pos in enumerate(players_data['position']):
                if pos in self.position_map:
                    position_onehot[i, self.position_map[pos]] = 1
                else:
                    print(f"Warning: Unknown position '{pos}', defaulting to MID")
                    position_onehot[i, self.position_map['MID']] = 1
            
            print(f"One-hot encoding shape: {position_onehot.shape}")
            return position_onehot
            
        except Exception as e:
            print(f"Error in position one-hot encoding: {str(e)}")
            print(f"Available columns: {players_data.columns.tolist()}")
            print(f"First few rows:\n{players_data.head()}")
            raise

    def _get_team_onehot(self, players_data: pd.DataFrame) -> np.ndarray:
        """
        获取球队的独热编码
        Args:
            players_data: 球员数据
        Returns:
            球队的独热编码矩阵
        """
        try:
            print("\nCreating team one-hot encoding...")
            print(f"Data shape: {players_data.shape}")
            print(f"Team distribution:\n{players_data['team'].value_counts()}")
            
            # 创建独热编码矩阵
            team_onehot = np.zeros((len(players_data), self.num_teams))
            
            # 为每个球员设置球队
            for i, team in enumerate(players_data['team']):
                if team in self.team_map:
                    team_onehot[i, self.team_map[team]] = 1
                else:
                    print(f"Warning: Unknown team '{team}'")
            
            print(f"Team one-hot shape: {team_onehot.shape}")
            return team_onehot
            
        except Exception as e:
            print(f"Error in team one-hot encoding: {str(e)}")
            print(f"Available columns: {players_data.columns.tolist()}")
            print(f"First few rows:\n{players_data.head()}")
            raise
    
    def _get_state(self):
        """获取当前状态"""
        # 获取当前比赛周的数据
        gw_data = self.data[self.data['gw'] == self.current_gameweek].copy()
        
        # 更新当前比赛周的球员映射
        current_players = gw_data['name'].unique()
        self.current_id_map = {name: idx for idx, name in enumerate(current_players)}
        self.current_inverse_map = {idx: name for name, idx in self.current_id_map.items()}
        
        # 提取特征
        features = self._extract_features(gw_data)
        
        return features
    
    def _extract_features(self, players_data):
        """提取特征"""
        # 确保输入数据是DataFrame
        if isinstance(players_data, pd.Series):
            players_data = pd.DataFrame([players_data])
        
        # 基本特征
        features = [
            players_data['value'].values,
            players_data['total_points'].values,
            players_data['minutes'].values,
            players_data['goals_scored'].values,
            players_data['assists'].values,
            players_data['clean_sheets'].values,
            players_data['goals_conceded'].values,
            players_data['own_goals'].values,
            players_data['penalties_saved'].values,
            players_data['penalties_missed'].values,
            players_data['yellow_cards'].values,
            players_data['red_cards'].values
        ]
        
        # 将所有特征堆叠成一个数组
        return np.stack(features, axis=-1)
    
    def _get_player_features(self, players_data: pd.DataFrame) -> np.ndarray:
        """
        获取球员特征
        Args:
            players_data: 球员数据
        Returns:
            球员特征矩阵
        """
        try:
            print("\nExtracting player features...")
            
            # 选择要使用的特征
            feature_cols = [
                'value',  # 价值
                'total_points',  # 总得分
                'minutes',  # 上场时间
                'goals_scored',  # 进球数
                'assists',  # 助攻数
                'clean_sheets',  # 零封数
                'goals_conceded',  # 丢球数
                'saves',  # 扑救数
                'bonus',  # 奖励分
                'bps',  # BPS
                'influence',  # 影响力
                'creativity',  # 创造力
                'threat',  # 威胁
                'ict_index'  # ICT指数
            ]
            
            # 确保所有特征列都存在
            for col in feature_cols:
                if col not in players_data.columns:
                    print(f"Warning: Feature {col} not found, using zeros")
                    players_data[col] = 0
            
            # 提取特征并转换为numpy数组
            features = players_data[feature_cols].values
            
            # 标准化特征
            features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
            
            print(f"Feature shape: {features.shape}")
            return features
            
        except Exception as e:
            print(f"Error extracting player features: {str(e)}")
            print(f"Available columns: {players_data.columns.tolist()}")
            print(f"First few rows:\n{players_data.head()}")
            raise

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
