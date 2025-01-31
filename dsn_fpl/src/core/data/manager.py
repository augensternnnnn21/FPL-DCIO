from typing import Dict, List, Optional
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from .collector import FPLDataCollector, RawPlayerData
from .preprocessor import DataPreprocessor
from ..temporal.origami_planner import TemporalState
import numpy as np

class DataManager:
    """数据管理器"""
    def __init__(self, data_path: Optional[Path] = None):
        """初始化数据管理器
        
        Args:
            data_path: 数据目录路径
        """
        self.collector = FPLDataCollector()
        self.preprocessor = DataPreprocessor()
        self.cache_duration = timedelta(minutes=30)
        self._cache = {}
        self._last_update = None
        self.data_path = data_path
        self.historical_data = self._load_data() if data_path else {}
        
    def _load_data(self) -> Dict:
        """从文件加载数据"""
        data = {}
        
        # 加载球员数据
        players_data = []
        seasons = [d for d in self.data_path.iterdir() if d.is_dir() and d.name.startswith('20')]
        
        for season_dir in seasons:
            season = season_dir.name
            players_raw = pd.read_csv(season_dir / 'players_raw.csv')
            
            # 处理球员名字
            if 'name' not in players_raw.columns:
                if 'first_name' in players_raw.columns and 'second_name' in players_raw.columns:
                    players_raw['name'] = players_raw['first_name'] + ' ' + players_raw['second_name']
                elif 'web_name' in players_raw.columns:
                    players_raw['name'] = players_raw['web_name']
                    
            # 处理球员ID
            if 'id' not in players_raw.columns and 'element' in players_raw.columns:
                players_raw['id'] = players_raw['element']
                
            # 处理球队ID
            if 'team' not in players_raw.columns and 'team_code' in players_raw.columns:
                players_raw['team'] = players_raw['team_code']
                
            # 处理位置
            if 'position' not in players_raw.columns and 'element_type' in players_raw.columns:
                position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                players_raw['position'] = players_raw['element_type'].map(position_map)
                
            players_raw['season'] = season
            players_data.append(players_raw)
            
        if players_data:
            data['players'] = pd.concat(players_data, ignore_index=True)
            
        # 加载赛程数据
        fixtures_data = {}
        for season_dir in seasons:
            season = season_dir.name
            fixtures_file = season_dir / 'fixtures.csv'
            if fixtures_file.exists():
                fixtures_data[season] = pd.read_csv(fixtures_file)
                
        if fixtures_data:
            data['fixtures'] = fixtures_data
            
        return data
        
    async def get_temporal_state(self) -> TemporalState:
        """获取当前时间维度状态"""
        # 检查缓存
        if (self._last_update and 
            datetime.now() - self._last_update < self.cache_duration):
            return self._cache.get('temporal_state')
            
        # 使用历史数据
        if not self.historical_data:
            raise ValueError("No historical data available")
            
        # 获取最新赛季的数据
        latest_season = self.historical_data['players']['season'].max()
        latest_data = self.historical_data['players'][
            self.historical_data['players']['season'] == latest_season
        ]
        
        # 获取活跃球员数据
        players_data = []
        for _, player in latest_data.iterrows():
            if player.get('minutes', 0) > 0:  # 只考虑有出场时间的球员
                player_data = RawPlayerData(
                    player_id=player['id'],
                    name=player['name'],
                    team=player['team'],
                    position=player['position'],
                    price=player.get('value', player.get('now_cost', 0)),
                    total_points=player.get('total_points', 0),
                    minutes=player.get('minutes', 0),
                    points=player.get('total_points', 0),
                    form=player.get('form', 0),
                    points_per_game=player.get('points_per_game', 0),
                    selected_by_percent=player.get('selected_by_percent', 0),
                    status='a'  # 假设所有球员都可用
                )
                players_data.append(player_data)
        
        # 预处理数据
        processed_data = {
            p.player_id: self.preprocessor.process_player_data(p)
            for p in players_data
        }
        
        # 获取赛程难度
        # 使用列表格式，每个元素代表未来一轮的难度（1-5，5最难）
        fixture_difficulty = {
            p.player_id: [
                {'difficulty': 3},  # 默认中等难度
                {'difficulty': 3},
                {'difficulty': 3}
            ]
            for p in players_data
        }
        
        # 如果有实际的赛程数据，可以在这里添加
        if 'fixtures' in self.historical_data:
            latest_fixtures = self.historical_data['fixtures'].get(latest_season)
            if latest_fixtures is not None:
                # TODO: 根据实际赛程数据更新fixture_difficulty
                pass
        
        # 创建时间维度状态
        temporal_state = self.preprocessor.create_temporal_state(
            processed_data,
            fixture_difficulty,
            datetime.now()
        )
        
        # 更新缓存
        self._cache['temporal_state'] = temporal_state
        self._last_update = datetime.now()
        
        return temporal_state

    def get_train_loader(self, batch_size: int = 32):
        """获取训练数据加载器"""
        # 确保数据已加载
        if not self.historical_data:
            raise ValueError("No historical data available")
            
        # 获取最新赛季的数据，排除验证集（20%）
        latest_season = self.historical_data['players']['season'].max()
        all_data = self.historical_data['players'][
            self.historical_data['players']['season'] == latest_season
        ]
        train_data = all_data.sample(frac=0.8, random_state=42)
        
        # 计算目标值的统计信息
        total_points_mean = train_data['total_points'].mean()
        total_points_std = train_data['total_points'].std()
        
        # 数据预处理
        processed_data = []
        for _, player in train_data.iterrows():
            if player.get('minutes', 0) > 0:  # 只考虑有出场时间的球员
                player_id = player['id']
                player_data = RawPlayerData(
                    player_id=player_id,
                    name=player['name'],
                    team=player['team'],
                    position=player['position'],
                    price=player.get('value', player.get('now_cost', 0)),
                    total_points=player.get('total_points', 0),
                    minutes=player.get('minutes', 0),
                    points=player.get('total_points', 0),
                    form=player.get('form', 0),
                    points_per_game=player.get('points_per_game', 0),
                    selected_by_percent=player.get('selected_by_percent', 0),
                    status='a'
                )
                features = self.preprocessor.process_player_data(player_data)
                
                # 构建基础统计数据
                stats = {
                    'minutes': player.get('minutes', 0),
                    'total_points': player.get('total_points', 0),
                    'goals_scored': player.get('goals_scored', 0),
                    'assists': player.get('assists', 0),
                    'clean_sheets': player.get('clean_sheets', 0),
                    'goals_conceded': player.get('goals_conceded', 0),
                    'own_goals': player.get('own_goals', 0),
                    'penalties_saved': player.get('penalties_saved', 0),
                    'penalties_missed': player.get('penalties_missed', 0),
                    'yellow_cards': player.get('yellow_cards', 0),
                    'red_cards': player.get('red_cards', 0),
                    'saves': player.get('saves', 0),
                    'bonus': player.get('bonus', 0),
                    'bps': player.get('bps', 0),
                    'influence': player.get('influence', 0),
                    'creativity': player.get('creativity', 0),
                    'threat': player.get('threat', 0),
                    'ict_index': player.get('ict_index', 0),
                    'selected_by_percent': player.get('selected_by_percent', 0),
                    'form': player.get('form', 0),
                    'fitness': 1.0,  # 默认值
                    'momentum': player.get('form', 0),  # 使用form作为momentum
                    'potential': 0.8  # 默认值
                }
                
                # 使用更稳定的目标值归一化方法
                total_points = player.get('total_points', 0)
                if total_points_std > 0:
                    normalized_target = (total_points - total_points_mean) / (total_points_std + 1e-8)
                    normalized_target = 0.5 * (1 + np.tanh(normalized_target / 2))  # 压缩到[0,1]范围
                else:
                    normalized_target = total_points / (total_points_mean + 1e-8)
                    normalized_target = np.clip(normalized_target, 0, 1)
                
                processed_data.append({
                    'features': features,
                    'targets': normalized_target,
                    'player_id': player_id,
                    'player_name': player['name'],
                    'position': player['position'],
                    'stats': stats,
                    'env_data': [],  # 空环境数据
                    'weather_data': {},  # 空天气数据
                    'media_data': [],  # 空媒体数据
                    'venue_data': {}  # 空场地数据
                })
        
        def batch_generator():
            while True:  # 无限循环生成数据
                import random
                data = processed_data.copy()
                random.shuffle(data)
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    batch_dict = {
                        'features': [item['features'] for item in batch],
                        'targets': [item['targets'] for item in batch],
                        'player_ids': [item['player_id'] for item in batch],
                        'player_names': {item['player_id']: item['player_name'] for item in batch},
                        'positions': {item['player_id']: item['position'] for item in batch},
                        'stats': {item['player_id']: item['stats'] for item in batch},
                        'env_data': [],  # 空环境数据
                        'weather_data': {},  # 空天气数据
                        'media_data': {},  # 空媒体数据
                        'venue_data': {}  # 空场地数据
                    }
                    yield batch_dict
                    
        return batch_generator()
        
    def get_val_loader(self, batch_size: int = 32):
        """获取验证数据加载器"""
        if not self.historical_data:
            raise ValueError("No historical data available")
            
        # 获取验证集（20%）
        latest_season = self.historical_data['players']['season'].max()
        all_data = self.historical_data['players'][
            self.historical_data['players']['season'] == latest_season
        ]
        val_data = all_data.sample(frac=0.2, random_state=42)
        
        # 计算目标值的统计信息
        total_points_mean = val_data['total_points'].mean()
        total_points_std = val_data['total_points'].std()
        
        # 数据预处理
        processed_data = []
        for _, player in val_data.iterrows():
            if player.get('minutes', 0) > 0:
                player_id = player['id']
                player_data = RawPlayerData(
                    player_id=player_id,
                    name=player['name'],
                    team=player['team'],
                    position=player['position'],
                    price=player.get('value', player.get('now_cost', 0)),
                    total_points=player.get('total_points', 0),
                    minutes=player.get('minutes', 0),
                    points=player.get('total_points', 0),
                    form=player.get('form', 0),
                    points_per_game=player.get('points_per_game', 0),
                    selected_by_percent=player.get('selected_by_percent', 0),
                    status='a'
                )
                features = self.preprocessor.process_player_data(player_data)
                
                # 构建基础统计数据
                stats = {
                    'minutes': player.get('minutes', 0),
                    'total_points': player.get('total_points', 0),
                    'goals_scored': player.get('goals_scored', 0),
                    'assists': player.get('assists', 0),
                    'clean_sheets': player.get('clean_sheets', 0),
                    'goals_conceded': player.get('goals_conceded', 0),
                    'own_goals': player.get('own_goals', 0),
                    'penalties_saved': player.get('penalties_saved', 0),
                    'penalties_missed': player.get('penalties_missed', 0),
                    'yellow_cards': player.get('yellow_cards', 0),
                    'red_cards': player.get('red_cards', 0),
                    'saves': player.get('saves', 0),
                    'bonus': player.get('bonus', 0),
                    'bps': player.get('bps', 0),
                    'influence': player.get('influence', 0),
                    'creativity': player.get('creativity', 0),
                    'threat': player.get('threat', 0),
                    'ict_index': player.get('ict_index', 0),
                    'selected_by_percent': player.get('selected_by_percent', 0),
                    'form': player.get('form', 0),
                    'fitness': 1.0,  # 默认值
                    'momentum': player.get('form', 0),  # 使用form作为momentum
                    'potential': 0.8  # 默认值
                }
                
                # 使用更稳定的目标值归一化方法
                total_points = player.get('total_points', 0)
                if total_points_std > 0:
                    normalized_target = (total_points - total_points_mean) / (total_points_std + 1e-8)
                    normalized_target = 0.5 * (1 + np.tanh(normalized_target / 2))  # 压缩到[0,1]范围
                else:
                    normalized_target = total_points / (total_points_mean + 1e-8)
                    normalized_target = np.clip(normalized_target, 0, 1)
                
                processed_data.append({
                    'features': features,
                    'targets': normalized_target,
                    'player_id': player_id,
                    'player_name': player['name'],
                    'position': player['position'],
                    'stats': stats,
                    'env_data': [],  # 空环境数据
                    'weather_data': {},  # 空天气数据
                    'media_data': [],  # 空媒体数据
                    'venue_data': {}  # 空场地数据
                })
        
        def batch_generator():
            while True:  # 无限循环生成数据
                for i in range(0, len(processed_data), batch_size):
                    batch = processed_data[i:i + batch_size]
                    batch_dict = {
                        'features': [item['features'] for item in batch],
                        'targets': [item['targets'] for item in batch],
                        'player_ids': [item['player_id'] for item in batch],
                        'player_names': {item['player_id']: item['player_name'] for item in batch},
                        'positions': {item['player_id']: item['position'] for item in batch},
                        'stats': {item['player_id']: item['stats'] for item in batch},
                        'env_data': [],  # 空环境数据
                        'weather_data': {},  # 空天气数据
                        'media_data': {},  # 空媒体数据
                        'venue_data': {}  # 空场地数据
                    }
                    yield batch_dict
                    
        return batch_generator()