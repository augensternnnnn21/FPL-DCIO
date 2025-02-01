import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os
import logging

class FPLDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        
        # 加载数据
        print(f"加载{split}数据...")
        self.data = self._load_data()
        
        # 预处理数据分组 - 优化分组过程
        print(f"预处理{split}数据分组...")
        # 1. 先按season和name进行排序
        self.data = self.data.sort_values(['season', 'name', 'gameweek'])
        
        # 2. 使用字典推导式一次性创建分组
        self.grouped_data = {
            (season, name): group.copy() 
            for (season, name), group in self.data.groupby(['season', 'name'])
        }
        
        print(f"{split}数据分组完成，共{len(self.grouped_data)}个分组")
        
        # 特征标准化
        self.scalers = self._fit_scalers() if split == 'train' else None
        self._transform_features()
        
    def _load_data(self):
        """加载原始数据"""
        # 主数据文件在 saman_project/data/processed/ 下
        data_path = os.path.join('saman_project', 'data', 'processed', f'{self.split}.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据文件: {data_path}。请先运行预处理脚本: python saman_project/utils/preprocess.py")
        
        print(f"加载{self.split}数据...")
        df = pd.read_csv(data_path)
        
        # 确保必要的列存在
        required_columns = ['name', 'season', 'gameweek', 'total_points']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据文件缺少必要的列: {missing_columns}")
        
        return df
    
    def _load_additional_data(self, df):
        """加载额外的数据源"""
        # 加载天气数据
        weather_path = os.path.join('saman_project', self.config.data_path, 'weather')
        weather_data = {}
        if os.path.exists(weather_path):
            for file in os.listdir(weather_path):
                if file.endswith('.json'):
                    with open(os.path.join(weather_path, file)) as f:
                        weather_data[file.split('.')[0]] = json.load(f)
        
        # 加载新闻数据
        news_path = os.path.join('saman_project', self.config.data_path, 'news')
        news_data = {}
        if os.path.exists(news_path):
            for file in os.listdir(news_path):
                if file.endswith('.json'):
                    with open(os.path.join(news_path, file)) as f:
                        news_data[file.split('.')[0]] = json.load(f)
        
        # 将额外数据添加到DataFrame
        self._merge_additional_data(df, weather_data, news_data)
    
    def _merge_additional_data(self, df, weather_data, news_data):
        """合并额外数据到主数据框"""
        # 添加天气特征
        if weather_data:
            df['temperature'] = df.apply(
                lambda x: weather_data.get(x.get('match_id', ''), {}).get('temperature', 0),
                axis=1
            )
            df['humidity'] = df.apply(
                lambda x: weather_data.get(x.get('match_id', ''), {}).get('humidity', 0),
                axis=1
            )
        else:
            df['temperature'] = 0
            df['humidity'] = 0
        
        # 添加新闻情感得分
        if news_data:
            df['news_sentiment'] = df.apply(
                lambda x: news_data.get(x.get('match_id', ''), {}).get('sentiment_score', 0),
                axis=1
            )
        else:
            df['news_sentiment'] = 0
    
    def _fit_scalers(self):
        """拟合特征标准化器"""
        scalers = {}
        for feature_type, columns in self._get_feature_columns().items():
            if columns:  # 如果特征列不为空
                # 1. 处理无效值
                data = self.data[columns].copy()
                data = data.fillna(0)  # 填充空值
                data = data.replace([np.inf, -np.inf], 0)  # 处理无穷值
                
                # 2. 处理异常值
                for col in columns:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    data[col] = data[col].clip(lower_bound, upper_bound)
                
                # 3. 标准化
                scaler = StandardScaler()
                scaler.fit(data)
                scalers[feature_type] = scaler
        
        # 为目标值（得分）创建单独的标准化器
        target_data = self.data['total_points'].values.reshape(-1, 1)
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(target_data)
        
        return scalers
    
    def inverse_transform_target(self, standardized_values):
        """将标准化的目标值转换回原始分数"""
        if isinstance(standardized_values, torch.Tensor):
            standardized_values = standardized_values.numpy()
        
        if len(standardized_values.shape) == 1:
            standardized_values = standardized_values.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(standardized_values).flatten()
    
    def _transform_features(self):
        """转换特征"""
        if self.split != 'train':
            # 加载特征和目标的标准化器
            saved_data = torch.load(os.path.join(self.config.save_path, 'scalers.pth'))
            self.scalers = saved_data['feature_scalers']
            self.target_scaler = saved_data['target_scaler']
        
        for feature_type, columns in self._get_feature_columns().items():
            if columns:  # 如果特征列不为空
                # 1. 处理无效值
                data = self.data[columns].copy()
                data = data.fillna(0)
                data = data.replace([np.inf, -np.inf], 0)
                
                # 2. 处理异常值
                for col in columns:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    data[col] = data[col].clip(lower_bound, upper_bound)
                
                # 3. 标准化
                self.data[columns] = self.scalers[feature_type].transform(data)
        
        if self.split == 'train':
            # 保存特征和目标的标准化器
            torch.save({
                'feature_scalers': self.scalers,
                'target_scaler': self.target_scaler
            }, os.path.join(self.config.save_path, 'scalers.pth'))
    
    def _get_feature_columns(self):
        """获取不同类型的特征列"""
        return {
            # 1. 历史策略特征
            'historical_strategy': [
                # 球员表现
                'total_points', 'minutes', 'goals_scored', 'assists',
                'clean_sheets', 'goals_conceded', 'own_goals',
                'penalties_saved', 'penalties_missed', 'yellow_cards',
                'red_cards', 'saves', 'bonus', 'bps',
                # 高级统计
                'influence', 'creativity', 'threat', 'ict_index',
                # 转会数据
                'transfers_balance', 'transfers_in', 'transfers_out',
                'selected', 'value'
            ],
            
            # 2. 当前环境特征
            'current_context': [
                # 比赛信息
                'opponent_team', 'was_home',
                # 球员状态
                'minutes_ma5', 'form', 'total_points_ma5',
                # 球队状态 (从比赛数据中获取)
                'team_h_score', 'team_a_score'  # 使用历史比分作为球队实力的代理指标
            ]
        }
    
    def _load_historical_data(self, current_gw, season, player_name):
        """加载历史数据 - 优化性能"""
        history = []
        player_season_data = self.grouped_data.get((season, player_name))
        
        if player_season_data is not None:
            # 使用布尔索引一次性获取过去5轮的数据
            mask = (player_season_data['gameweek'] >= max(1, current_gw - 5)) & (player_season_data['gameweek'] < current_gw)
            history_data = player_season_data[mask]
            
            # 按轮次分组并转换为列表
            history = [group for _, group in history_data.groupby('gameweek')]
        
        return history

    def _load_current_context(self, gameweek):
        """加载当前环境数据"""
        try:
            # fixtures.csv 在2023-24赛季目录下
            fixtures = pd.read_csv(os.path.join('data', '2023-24', 'fixtures.csv'))
            
            # 处理赛程数据中的轮次信息
            if 'event' not in fixtures.columns and 'gameweek' in fixtures.columns:
                fixtures['event'] = fixtures['gameweek']
            
            current_fixtures = fixtures[fixtures['gameweek'] == gameweek]
            
            # 加载球员最新状态
            try:
                latest_gw = pd.read_csv(os.path.join('data', '2023-24', 'gws', f'gw{gameweek-1}.csv'))
            except:
                latest_gw = pd.DataFrame()
            
            # 如果数据为空，创建带有默认值的DataFrame
            if latest_gw.empty:
                latest_gw = pd.DataFrame({
                    'name': [],
                    'minutes': [],
                    'total_points': [],
                    'form': []
                })
            
            return {
                'fixtures': current_fixtures,
                'player_state': latest_gw
            }
        except Exception as e:
            print(f"Warning: Error loading current context for GW{gameweek}: {str(e)}")
            # 返回空数据
            return {
                'fixtures': pd.DataFrame(),
                'player_state': pd.DataFrame()
            }

    def _calculate_form(self, history_data, player_name):
        """计算球员状态（只使用历史数据）"""
        recent_games = [
            df[df['name'] == player_name] for df in history_data
        ]
        
        if not recent_games or all(g.empty for g in recent_games):
            return {
                'form': 0,
                'minutes_last5': 0,
                'avg_points': 0,
                'trend': 0,
                'total_points_ma5': 0,
                'minutes_ma5': 0,
                'goals_scored_ma5': 0,
                'assists_ma5': 0,
                'clean_sheets_ma5': 0,
                'goals_conceded_ma5': 0,
                'own_goals_ma5': 0,
                'penalties_saved_ma5': 0,
                'penalties_missed_ma5': 0,
                'yellow_cards_ma5': 0,
                'red_cards_ma5': 0,
                'saves_ma5': 0,
                'bonus_ma5': 0,
                'bps_ma5': 0,
                'team_goals_scored_ma5': 0,
                'team_goals_conceded_ma5': 0
            }
        
        # 计算历史统计量
        stats = {}
        for stat in ['total_points', 'minutes', 'goals_scored', 'assists',
                     'clean_sheets', 'goals_conceded', 'own_goals',
                     'penalties_saved', 'penalties_missed', 'yellow_cards',
                     'red_cards', 'saves', 'bonus', 'bps']:
            values = []
            for game in recent_games:
                if not game.empty and stat in game.columns:
                    values.append(game[stat].iloc[0])
                else:
                    values.append(0)
            stats[f'{stat}_ma5'] = np.mean(values) if values else 0
        
        # 计算球队统计量
        team_stats = {}
        for stat in ['team_goals_scored', 'team_goals_conceded']:
            values = []
            for game in recent_games:
                if not game.empty and stat in game.columns:
                    values.append(game[stat].iloc[0])
                else:
                    values.append(0)
            team_stats[f'{stat}_ma5'] = np.mean(values) if values else 0
        
        # 计算总出场时间
        minutes_values = []
        for game in recent_games:
            if not game.empty and 'minutes' in game.columns:
                minutes_values.append(game['minutes'].iloc[0])
            else:
                minutes_values.append(0)
        total_minutes = sum(minutes_values)
        
        # 计算趋势
        points_values = []
        for game in recent_games:
            if not game.empty and 'total_points' in game.columns:
                points_values.append(game['total_points'].iloc[0])
            else:
                points_values.append(0)
        
        trend = 0
        if len(points_values) > 1:
            try:
                trend = np.polyfit(range(len(points_values)), points_values, 1)[0]
            except:
                trend = 0
        
        return {
            'form': stats['total_points_ma5'],
            'minutes_last5': total_minutes,
            'avg_points': stats['total_points_ma5'],
            'trend': trend,
            **stats,
            **team_stats
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        player_data = self.data.iloc[idx]
        gameweek = player_data['gameweek']
        season = player_data['season']
        player_name = player_data['name']
        
        # 1. 获取历史数据（不包括当前轮次）
        history_data = self._load_historical_data(gameweek, season, player_name)
        player_history = self._calculate_form(history_data, player_name)
        
        # 2. 获取当前环境（不包括当前轮次的比赛数据）
        current_context = self._load_current_context(gameweek)
        
        # 3. 构建特征字典（只使用历史数据）
        features = {
            'historical': {
                'performance': torch.FloatTensor([
                    player_history['form'],  # 使用历史form
                    player_history['minutes_last5'],
                    player_history['avg_points'],
                    player_history['trend']
                ]),
                'player_stats': torch.FloatTensor([
                    player_history['total_points_ma5'],  # 使用历史平均
                    player_history['minutes_ma5'],
                    player_history['goals_scored_ma5'],
                    player_history['assists_ma5'],
                    player_history['clean_sheets_ma5'],
                    player_history['goals_conceded_ma5'],
                    player_history['own_goals_ma5'],
                    player_history['penalties_saved_ma5'],
                    player_history['penalties_missed_ma5'],
                    player_history['yellow_cards_ma5'],
                    player_history['red_cards_ma5'],
                    player_history['saves_ma5'],
                    player_history['bonus_ma5'],
                    player_history['bps_ma5']
                ]),
                'transfers': torch.FloatTensor([
                    player_data.get('transfers_balance', 0),  # 使用get方法处理可能不存在的列
                    player_data.get('transfers_in', 0),
                    player_data.get('transfers_out', 0),
                    player_data.get('selected', 0),
                    player_data.get('value', 0)
                ])
            },
            'context': {
                'fixture': torch.FloatTensor([
                    player_data.get('opponent_team', 0),  # 使用get方法处理可能不存在的列
                    player_data.get('was_home', 0)
                ]),
                'player_state': torch.FloatTensor([
                    player_history['minutes_ma5'],  # 使用历史数据
                    player_history['form'],
                    player_history['avg_points']
                ]),
                'team_state': torch.FloatTensor([
                    player_history.get('team_goals_scored_ma5', 0),  # 使用get方法处理可能不存在的列
                    player_history.get('team_goals_conceded_ma5', 0)
                ])
            }
        }
        
        # 4. 获取标签（当前轮次的实际得分）
        target = torch.FloatTensor([player_data['total_points']])
        
        return features, target

    def _get_fixture_features(self, current_context, player_data):
        """提取比赛特征"""
        fixtures = current_context['fixtures']
        if fixtures.empty:
            return torch.zeros(2)  # [opponent_team, was_home]
        
        # 获取当前比赛
        current_fixture = fixtures.iloc[0]
        return torch.FloatTensor([
            current_fixture.get('team_a', 0),  # 对手球队
            1 if current_fixture.get('was_home', False) else 0  # 主场标志
        ])
    
    def _get_player_state_features(self, current_context, player_data):
        """提取球员状态特征"""
        player_state = current_context['player_state']
        if player_state.empty:
            return torch.zeros(3)  # [minutes_ma5, form, total_points_ma5]
        
        # 获取当前球员状态
        current_state = player_state.iloc[0]
        return torch.FloatTensor([
            current_state.get('minutes_ma5', 0),
            current_state.get('form', 0),
            current_state.get('total_points_ma5', 0)
        ])
    
    def _get_team_state_features(self, current_context, player_data):
        """提取球队状态特征"""
        teams = current_context['teams']
        if teams.empty:
            return torch.zeros(4)  # [strength_overall, strength_overall_home, strength_overall_away, opponent_strength]
        
        # 获取当前球队状态
        team_state = teams[teams['team_id'] == player_data.get('team', 0)].iloc[0] if not teams.empty else pd.Series()
        return torch.FloatTensor([
            team_state.get('team_strength', 0),
            player_data.get('opponent_strength', 0)
        ]) 