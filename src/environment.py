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
    def __init__(self, data_path: str, budget: float = 100.0, max_players: int = 15, history_length: int = 5, season=None):
        """
        初始化FPL环境
        Args:
            data_path: 数据路径
            budget: 预算限制
            max_players: 最大球员数量
            history_length: 历史数据长度
            season: 可选的赛季参数
        """
        super().__init__()
        print(f"Loading raw data from {data_path}")
        self.data_path = data_path
        self.budget = budget
        self.max_players = max_players
        self.history_length = history_length
        self.current_season = season
        
        # 位置映射
        self.position_map = {
            'GKP': 0,  # 门将
            'DEF': 1,  # 后卫
            'MID': 2,  # 中场
            'FWD': 3   # 前锋
        }
        self.num_positions = len(self.position_map)
        
        # 加载数据
        self.data = self._load_data()
        if self.data is None:
            raise ValueError(f"Failed to load data from {self.data_path}")
        
        # 添加球队信息
        self._add_team_info()
        
        # 获取所有球队
        self.teams = sorted(self.data['team'].unique())
        self.team_map = {team: idx for idx, team in enumerate(self.teams)}
        self.num_teams = len(self.teams)
        
        # 初始化状态变量
        self.current_gameweek = 0
        self.selected_players = []
        self.available_budget = budget
        self.history = []
        
        # 约束参数
        self.position_limits = {
            'GK': 2,
            'DEF': 5,
            'MID': 5,
            'FWD': 3
        }
        self.max_per_team = 3
        
        # 初始化状态
        self.reset()
    
    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        print(f"\nDEBUG: Loading raw data from {self.data_path}")
        try:
            if self.current_season:
                season_path = os.path.join(self.data_path, self.current_season)
                print(f"Loading processed data from {season_path}")
                # 检查是否存在处理好的数据文件
                processed_file = os.path.join(season_path, f"{self.current_season}_processed.csv")
                if os.path.exists(processed_file):
                    print(f"Loading processed data from {processed_file}")
                    return pd.read_csv(processed_file)
                
                print(f"Loading raw data from {season_path}/gws")
                all_data = []
                gw_path = os.path.join(season_path, 'gws')
                
                if not os.path.exists(gw_path):
                    raise ValueError(f"Game week directory not found: {gw_path}")
                
                # 加载每个比赛周的数据
                for gw in range(1, 39):  # 假设有38个比赛周
                    gw_file = os.path.join(gw_path, f'gw{gw}.csv')
                    if os.path.exists(gw_file):
                        try:
                            # 尝试不同的编码
                            for encoding in ['utf-8', 'latin1', 'cp1252']:
                                try:
                                    df = pd.read_csv(gw_file, encoding=encoding)
                                    df['gameweek'] = gw
                                    all_data.append(df)
                                    print(f"Successfully loaded gameweek {gw} with {encoding} encoding")
                                    break
                                except UnicodeDecodeError:
                                    continue
                        except Exception as e:
                            print(f"Error loading gameweek {gw}: {e}")
                
                if not all_data:
                    raise ValueError(f"No gameweek data loaded from: {season_path}")
                
                # 合并所有数据
                data = pd.concat(all_data, ignore_index=True)
                
                # 添加位置信息
                data = self._infer_position(data)
                
                # 保存处理后的数据
                data.to_csv(processed_file, index=False)
                print(f"Saved processed data to {processed_file}")
                
                return data
            else:
                # 如果没有指定赛季，使用默认路径
                if os.path.exists(self.data_path):
                    return pd.read_csv(self.data_path)
                else:
                    raise ValueError(f"Data file not found: {self.data_path}")
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
        try:
            # 获取当前可用的球员
            available_players = self._get_available_players()
            
            # 获取球员特征
            player_features = self._get_player_features(available_players)
            
            # 获取位置的独热编码
            position_onehot = self._get_position_onehot(available_players)
            
            # 获取球队的独热编码
            team_onehot = self._get_team_onehot(available_players)
            
            # 组合所有特征
            state = {
                'player_features': player_features,
                'position_onehot': position_onehot,
                'team_onehot': team_onehot,
                'budget': self.budget,
                'current_players': len(self.selected_players),
                'gameweek': self.current_gameweek
            }
            
            return state
            
        except Exception as e:
            print(f"Error getting state: {str(e)}")
            print(f"Current gameweek: {self.current_gameweek}")
            print(f"Selected players: {self.selected_players}")
            raise
    
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
        """执行动作"""
        # 获取当前比赛周的数据
        gw_data = self.data[self.data['gw'] == self.current_gameweek].copy()
        
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
            (self.data['gw'] == self.current_gameweek) & 
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
        position_reward = sum([1 if positions.get(pos, 0) == req else -0.2 for pos, req in self.position_limits.items()])
        budget_reward = 0.5 * (total_cost / BUDGET)  # 鼓励使用预算

        # 新增：对手强度惩罚（对手过去5场平均失球数越少，得分难度越高）
        team_strength = selected_players['team_strength'].mean()
        strength_bonus = 0.2 * team_strength  # 使用球队强度作为奖励组件
        
        # 计算得分奖励
        total_points = selected_players['total_points'].sum()
        
        # 处理第一周的特殊情况
        if self.current_gameweek == 1:
            current_max_points = self.data[self.data['gw'] == 1]['total_points'].max()
        else:
            current_max_points = self.data[self.data['gw'] == self.current_gameweek]['total_points'].max()
        
        # 添加阵容稳定性约束
        stability_penalty = 0
        if self.current_gameweek > 1:
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
            print(f"本周最高分: {self.max_points_by_gw.loc[self.current_gameweek]:.1f}")
        
        # 更新状态
        self.selected_players = valid_actions
        self.budget = self.budget - total_cost
        
        # 更新比赛周
        self.current_gameweek += 1
        done = self.current_gameweek > 38
        
        # 确保返回的状态是字典
        next_state = self._get_state()
        return next_state, reward, done, False, {}

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
