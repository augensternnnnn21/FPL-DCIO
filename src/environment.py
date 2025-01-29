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
from utils.logger import FPLLogger


class FPLEnv(gym.Env):
    """FPL环境"""
    def __init__(self, config):
        """初始化FPL环境
        Args:
            config: 配置对象，包含以下字段：
                - data_path: 数据路径
                - train_seasons: 训练赛季列表
                - val_season: 验证赛季
                - test_season: 测试赛季
                - budget: 预算限制
                - max_players: 最大球员数量
        """
        super().__init__()
        
        self.config = config
        self.data_path = config.data_path
        self.train_seasons = config.train_seasons
        self.val_season = config.val_season
        self.test_season = config.test_season
        self.budget = config.budget
        self.max_players = config.max_players
        
        # 初始化logger
        self.logger = FPLLogger(os.path.join(config.output_path, 'env_logs'))
        
        # 直接使用具体的文件路径
        self.base_dir = Path(config.data_path)
        self.seasons_file = self.base_dir / "cleaned_merged_seasons.csv"
        self.teams_file = self.base_dir / "master_team_list.csv"
        
        # 验证数据文件
        self._validate_data_files()
        
        # 初始化组件
        self.preprocessor = FPLPreprocessor()
        
        # 初始化映射
        self.position_mapping = {
            'GK': 0,
            'DEF': 1,
            'MID': 2,
            'FWD': 3
        }
        
        try:
            # 初始化数据加载器并加载数据
            self.data_loader = FPLDataLoader(
                seasons_path=self.seasons_file,
                teams_path=self.teams_file
            )
            raw_data = self.data_loader.load_data()
            self.team_data = self.data_loader.load_team_list()
            
            # 预处理数据
            self.data = self.preprocessor.preprocess(raw_data, self.team_data)
            
            # 获取可用赛季和球员
            self.available_seasons = np.unique(self.data['season'].values)
            self.all_players = self.data['name'].unique()
            
            self.logger.info(f"Successfully loaded {len(self.data)} records")
            self.logger.info(f"Available seasons: {self.available_seasons}")
            self.logger.info(f"Total players: {len(self.all_players)}")
            
            # 处理球队映射，确保所有值都是字符串类型
            teams = self.data['team'].unique()
            teams = [str(t) for t in teams if pd.notna(t)]  # 过滤掉 NaN 并转换为字符串
            self.team_mapping = {team: i for i, team in enumerate(sorted(teams))}
            
        except Exception as e:
            self.logger.error(f"Data loading error: {str(e)}")
            raise
            
        # 保存参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 验证数据
        self._validate_data()
        
        # 初始化空间
        self._init_spaces()
        
        # 初始化状态
        self.current_gameweek = 1
        self.selected_players = []
        self.total_cost = 0
        self.violations = 0
        
        # 特殊卡片标记
        self.special_cards = {
            'triple_captain': False,
            'bench_boost': False
        }
        
        # 预计算一些常用值以提高性能
        try:
            # 预计算特征统计
            numerical_columns = ['value', 'total_points', 'minutes', 'goals_scored',
                               'assists', 'clean_sheets', 'goals_conceded']
            
            self.feature_stats = {}
            for col in numerical_columns:
                if col in self.data.columns:
                    values = self.data[col].dropna()  # 去除NaN值
                    if len(values) > 0:
                        self.feature_stats[col] = {
                            'mean': float(values.mean()),
                            'std': float(values.std() or 1.0)  # 如果std为0则使用1
                        }
                    else:
                        self.logger.warning(f"Column {col} is empty after dropping NaN values")
                        self.feature_stats[col] = {'mean': 0.0, 'std': 1.0}
                else:
                    self.logger.warning(f"Column {col} not found in data")
                    self.feature_stats[col] = {'mean': 0.0, 'std': 1.0}
                
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            self.logger.error(f"Data columns: {self.data.columns.tolist()}")
            self.logger.error(f"Data types: {self.data.dtypes}")
            raise
    
    def _validate_data(self):
        """验证数据完整性"""
        try:
            # 标准化列名
            column_mapping = {
                'element': 'player_name',
                'element_type': 'pos',
                'round': 'GW',
                'web_name': 'player_name',
                'name': 'player_name',
                'value': 'price',
                'total_points': 'points',
                'position': 'pos'
            }
            
            # 重命名列
            self.data = self.data.rename(columns=column_mapping)
            
            # 必需的列
            required_columns = [
                'player_name',     # 球员名
                'pos',            # 位置
                'team',           # 球队
                'price',          # 价值
                'points',         # 总得分
                'minutes',        # 出场时间
                'goals_scored',   # 进球
                'assists',        # 助攻
                'clean_sheets',   # 零封
                'goals_conceded', # 丢球
                'season',         # 赛季
                'GW'             # 比赛周
            ]
            
            # 检查必需的列是否存在
            missing_cols = [col for col in required_columns if col not in self.data.columns]
            if missing_cols:
                self.logger.error(f"Available columns: {self.data.columns.tolist()}")
                raise ValueError(f"缺少必需的列: {missing_cols}")
            
            # 检查数值列
            numeric_columns = [
                'price', 'points', 'minutes', 'goals_scored', 
                'assists', 'clean_sheets', 'goals_conceded'
            ]
            
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    self.data[col].fillna(0, inplace=True)
            
            # 检查分类列
            categorical_columns = ['pos', 'team', 'season']
            for col in categorical_columns:
                if not pd.api.types.is_categorical_dtype(self.data[col]):
                    self.data[col] = self.data[col].astype('category')
            
            # 检查时间列
            if not pd.api.types.is_integer_dtype(self.data['GW']):
                self.data['GW'] = pd.to_numeric(self.data['GW'], errors='coerce').astype(int)
            
            # 验证数据范围
            if self.data['price'].min() < 0 or self.data['price'].max() > 200:
                raise ValueError("球员价值超出合理范围")
            
            if self.data['points'].min() < -20 or self.data['points'].max() > 200:
                raise ValueError("球员得分超出合理范围")
            
            if self.data['minutes'].min() < 0 or self.data['minutes'].max() > 90:
                raise ValueError("出场时间超出合理范围")
            
            # 记录基本统计信息
            self.logger.info(f"Successfully loaded {len(self.data)} records")
            self.logger.info(f"Available seasons: {self.data['season'].unique()}")
            self.logger.info(f"Total players: {self.data['player_name'].nunique()}")
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {str(e)}")
            self.logger.error(f"Data shape: {self.data.shape}")
            self.logger.error(f"Available columns: {self.data.columns.tolist()}")
            raise

    def _init_spaces(self):
        """初始化动作和观察空间"""
        try:
            # 获取第一个游戏周的数据来确定维度
            first_week_data = self.data[self.data['gw'] == 1]
            n_players = len(first_week_data)
            
            # 动作空间：每个球员是否被选中（二值）
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(n_players,),
                dtype=np.int32
            )
            
            # 观察空间：球员特征 + 位置编码 + 球队编码
            n_features = 12  # 基础特征数量
            n_positions = len(self.position_mapping)  # 位置数量
            n_teams = len(self.team_mapping)  # 球队数量
            
            self.observation_space = spaces.Dict({
                'player_features': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_players, n_features),
                    dtype=np.float32
                ),
                'position_onehot': spaces.Box(
                    low=0,
                    high=1,
                    shape=(n_players, n_positions),
                    dtype=np.int32
                ),
                'team_onehot': spaces.Box(
                    low=0,
                    high=1,
                    shape=(n_players, n_teams),
                    dtype=np.int32
                ),
                'gameweek': spaces.Discrete(39),  # 1-38周
                'raw_features': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_players, n_features),
                    dtype=np.float32
                )
            })
            
            self.logger.info(f"Feature dimensions: base={n_features}, position={n_positions}, team={n_teams}")
            
        except Exception as e:
            self.logger.error(f"Error initializing spaces: {str(e)}")
            raise
    
    def reset(self, season=None):
        """重置环境"""
        try:
            # 选择赛季
            if season is None:
                season = np.random.choice(self.available_seasons)
            
            # 加载数据
            self.current_data = self._load_season_data(season)
            self.current_gameweek = 1
            self.selected_players = []
            self.total_cost = 0
            self.violations = 0
            
            # 获取当前状态
            state = self._get_state()
            
            return state
        except Exception as e:
            self.logger.error(f"Error in reset: {str(e)}")
            raise

    def _load_season_data(self, season):
        """加载赛季数据"""
        try:
            # 创建完整的副本
            season_data = self.data[self.data['season'] == season].copy(deep=True)
            
            if len(season_data) == 0:
                self.logger.warning(f"未找到 {season} 赛季的数据")
                closest_season = self._find_closest_season(season)
                self.logger.warning(f"使用最接近的赛季 {closest_season} 作为替代")
                season_data = self.data[self.data['season'] == closest_season].copy(deep=True)
            
            # 预处理数据
            season_data.loc[:, 'team'] = season_data['team'].astype(str)
            
            return season_data
            
        except Exception as e:
            self.logger.error(f"Error loading season data: {str(e)}")
            self.logger.error(f"Available columns: {self.data.columns.tolist()}")
            raise

    def _find_closest_season(self, target_season: str) -> str:
        """找到最接近的赛季"""
        available_seasons = self.data['season'].unique()
        target_year = int(target_season.split('-')[0])
        
        closest_season = min(
            available_seasons,
            key=lambda x: abs(int(x.split('-')[0]) - target_year)
        )
        
        return closest_season

    def step(self, action):
        """执行一步动作"""
        try:
            # 获取当前数据并创建副本
            current_data = self.data[self.data['gw'] == self.current_gameweek].copy(deep=True)
            
            # 预处理数据
            current_data.loc[:, 'team'] = current_data['team'].astype(str)
            
            # 确保动作维度与当前数据匹配
            if len(action) != len(current_data):
                if len(action) > len(current_data):
                    self.logger.warning(f"Truncating action from {len(action)} to {len(current_data)}")
                    action = action[:len(current_data)]
                else:
                    self.logger.warning(f"Padding action from {len(action)} to {len(current_data)}")
                    action = np.pad(action, (0, len(current_data) - len(action)))
            
            # 计算奖励
            reward, info = self._calculate_reward(action, current_data)
            
            # 更新游戏周
            self.current_gameweek += 1
            
            # 检查是否结束
            done = self.current_gameweek > 38
            
            # 获取下一个状态
            next_state = self._get_state() if not done else None
            
            return next_state, reward, done, False, info
            
        except Exception as e:
            self.logger.error(f"Error in step: {str(e)}")
            return self._get_state(), 0, True, False, {}

    def _check_budget_constraint(self, action, current_data):
        """检查预算约束"""
        try:
            # 使用 'price' 而不是 'value'
            if 'price' not in current_data.columns:
                self.logger.error("Price column not found in data")
                return 0
            
            total_cost = np.sum(current_data['price'].values * action)
            if total_cost > self.budget:
                return total_cost - self.budget
            return 0
        
        except Exception as e:
            self.logger.error(f"Error in budget constraint check: {str(e)}")
            return 0

    def _check_squad_size_constraint(self, action):
        """检查球队规模约束"""
        num_selected = np.sum(action)
        return abs(num_selected - self.max_players)

    def _check_position_constraint(self, action, current_data):
        """检查位置约束"""
        position_counts = {
            'GK': np.sum(action * (current_data['pos'] == 'GK')),
            'DEF': np.sum(action * (current_data['pos'] == 'DEF')),
            'MID': np.sum(action * (current_data['pos'] == 'MID')),
            'FWD': np.sum(action * (current_data['pos'] == 'FWD'))
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
            # 定义基础特征和它们的映射
            feature_mapping = {
                'price': ['price', 'value'],          # 价值
                'points': ['points', 'total_points'], # 得分
                'minutes': ['minutes'],               # 出场时间
                'goals_scored': ['goals_scored'],     # 进球
                'assists': ['assists'],               # 助攻
                'clean_sheets': ['clean_sheets'],     # 零封
                'goals_conceded': ['goals_conceded'], # 丢球
                'xP': ['xP'],                        # 预期得分
                'expected_goals': ['expected_goals'], # 预期进球
                'expected_assists': ['expected_assists']# 预期助攻
            }
            
            # 收集特征值
            feature_values = []
            for feature, possible_names in feature_mapping.items():
                # 尝试所有可能的列名
                found = False
                for col_name in possible_names:
                    if col_name in current_data.columns:
                        values = current_data[col_name].values
                        found = True
                        break
                if not found:
                    self.logger.warning(f"Missing feature: {feature}")
                    values = np.zeros(len(current_data))
                feature_values.append(values)
            
            # 转换为numpy数组
            features_array = np.array(feature_values).T
            
            # 标准化特征
            features_array = (features_array - np.mean(features_array, axis=0)) / (np.std(features_array, axis=0) + 1e-8)
            
            # 添加计算的特征
            points_col = 'points' if 'points' in current_data.columns else 'total_points'
            price_col = 'price' if 'price' in current_data.columns else 'value'
            
            if 'minutes' in current_data.columns and points_col in current_data.columns:
                points_per_minute = current_data.apply(
                    lambda x: x[points_col] / x['minutes'] if x['minutes'] > 0 else 0,
                    axis=1
                ).values
                features_array = np.column_stack([features_array, points_per_minute])
            
            if price_col in current_data.columns and points_col in current_data.columns:
                roi = (current_data[points_col] / current_data[price_col]).values
                features_array = np.column_stack([features_array, roi])
            
            return features_array
            
        except Exception as e:
            self.logger.error(f"Error in _get_state_features: {str(e)}")
            self.logger.error(f"Current data columns: {current_data.columns}")
            raise

    def _get_position_onehot(self, current_data):
        """获取位置的one-hot编码"""
        try:
            n_players = len(current_data)
            n_positions = len(self.position_mapping)
            position_onehot = np.zeros((n_players, n_positions))
            
            # 使用 'pos' 而不是 'position'
            for i, pos in enumerate(current_data['pos']):
                if pos in self.position_mapping:
                    position_onehot[i, self.position_mapping[pos]] = 1
            
            return position_onehot
            
        except Exception as e:
            self.logger.error(f"Error in _get_position_onehot: {str(e)}")
            self.logger.error(f"Available columns: {current_data.columns.tolist()}")
            self.logger.error(f"Position values: {current_data['pos'].unique() if 'pos' in current_data.columns else 'pos column not found'}")
            raise
    
    def _get_team_onehot(self, current_data):
        """获取球队的one-hot编码"""
        try:
            # 确保所有team值都是字符串类型，并过滤掉NaN
            teams = current_data['team'].dropna().astype(str).unique()
            teams = sorted(teams)  # 现在可以安全排序
            
            # 创建one-hot编码
            onehot = np.zeros((len(current_data), len(teams)))
            for i, team in enumerate(teams):
                onehot[:, i] = (current_data['team'].astype(str) == team).astype(float)
            
            # 记录日志
            self.logger.debug(f"Created team one-hot encoding with shape: {onehot.shape}")
            
            return onehot
            
        except Exception as e:
            self.logger.error(f"Error in _get_team_onehot: {str(e)}")
            self.logger.error(f"Team values: {current_data['team'].value_counts().to_dict()}")
            self.logger.error(f"Team dtypes: {current_data['team'].dtype}")
            raise

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
            # 初始化信息字典
            info = {
                'violations': 0,
                'team_value': 0,
                'points': 0
            }
            
            # 1. 基础奖励：选中球员的得分总和
            if 'points' in current_data.columns:
                points = np.sum(current_data['points'].values * action)
                info['points'] = points
                base_reward = points
            else:
                base_reward = 0
            
            # 2. 预算约束惩罚
            budget_violation = self._check_budget_constraint(action, current_data)
            if budget_violation > 0:
                info['violations'] += 1
                base_reward -= budget_violation * 0.5  # 惩罚系数可调
            
            # 3. 球队规模约束惩罚
            size_violation = self._check_squad_size_constraint(action)
            if size_violation > 0:
                info['violations'] += 1
                base_reward -= size_violation * 0.5
            
            # 4. 记录球队价值 (使用 'price' 而不是 'value')
            if 'price' in current_data.columns:
                team_value = np.sum(current_data['price'].values * action)
                info['team_value'] = team_value
            
            # 5. 添加协同奖励
            synergy_matrix = self._build_synergy_matrix(current_data)
            synergy_reward = np.sum(synergy_matrix * np.outer(action, action))
            
            total_reward = base_reward + synergy_reward
            
            self.logger.debug(f"Reward breakdown - Base: {base_reward:.2f}, Synergy: {synergy_reward:.2f}")
            
            return total_reward, info
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            self.logger.error(f"Available columns: {current_data.columns.tolist()}")  # 添加列名日志
            return 0, {'violations': 0, 'team_value': 0, 'points': 0}

    def _get_state(self):
        """获取当前状态"""
        try:
            current_data = self.data[self.data['gw'] == self.current_gameweek].copy(deep=True)
            
            if len(current_data) == 0:
                raise ValueError(f"No data found for gameweek {self.current_gameweek}")
            
            # 确保team列的数据类型一致性
            current_data.loc[:, 'team'] = current_data['team'].astype(str)
            
            # 标准化列名映射
            column_mapping = {
                'player_name': 'name',
                'round': 'gw',
                'value': 'price',
                'total_points': 'points'
            }
            
            # 重命名列
            current_data = current_data.rename(columns=column_mapping)
            
            # 打印当前可用的列
            self.logger.debug(f"Available columns after mapping: {current_data.columns.tolist()}")
            
            # 检查必要的列
            required_columns = [
                'name',          # 原 player_name
                'pos',          # 保持使用 pos
                'team',
                'price',        # 原 value
                'points',       # 原 total_points
                'minutes',
                'goals_scored',
                'assists',
                'clean_sheets',
                'goals_conceded',
                'gw'           # 原 GW/round
            ]
            
            missing_cols = [col for col in required_columns if col not in current_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 生成特征
            state = {
                'player_features': self._get_state_features(current_data),
                'position_onehot': self._get_position_onehot(current_data),
                'team_onehot': self._get_team_onehot(current_data),
                'gameweek': self.current_gameweek,
                'raw_features': None  # 将在下面设置
            }
            
            # 保存原始特征
            state['raw_features'] = state['player_features'].copy()
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in _get_state: {str(e)}")
            self.logger.error(f"Current gameweek: {self.current_gameweek}")
            self.logger.error(f"Data shape: {self.data.shape}")
            self.logger.error(f"Data columns: {current_data.columns.tolist()}")
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

    def _update_team_data(self, current_data):
        """更新球队数据"""
        try:
            # 创建完整的副本而不是视图
            current_data = current_data.copy(deep=True)
            
            # 使用 loc 进行赋值
            current_data.loc[:, 'team'] = current_data['team'].astype(str)
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error updating team data: {str(e)}")
            return current_data