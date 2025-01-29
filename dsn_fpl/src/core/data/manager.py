from typing import Dict, List, Optional
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from .collector import FPLDataCollector, RawPlayerData
from .preprocessor import DataPreprocessor
from ..temporal.origami_planner import TemporalState

class DataManager:
    """数据管理器"""
    def __init__(self, historical_data: Dict = None):
        self.collector = FPLDataCollector()
        self.preprocessor = DataPreprocessor()
        self.cache_duration = timedelta(minutes=30)
        self._cache = {}
        self._last_update = None
        self.historical_data = historical_data or {}
        
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
            if player['minutes'] > 0:  # 只考虑有出场时间的球员
                player_data = RawPlayerData(
                    player_id=player['id'],
                    name=player['name'],
                    team=player['team'],
                    position=player['position'],
                    price=player.get('value', 0),
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