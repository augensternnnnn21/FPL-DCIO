from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ..temporal.origami_planner import TemporalState
from .collector import RawPlayerData

class DataPreprocessor:
    """数据预处理器"""
    def __init__(self):
        self.form_window = 5  # 状态计算窗口
        self.price_window = 3  # 价格趋势窗口
        
    def process_player_data(self, raw_data: RawPlayerData) -> Dict[str, np.ndarray]:
        """处理原始球员数据"""
        # 计算状态轨迹
        form_trajectory = self._compute_form_trajectory(
            raw_data.points,
            raw_data.minutes
        )
        
        # 计算疲劳指数
        fatigue = self._compute_fatigue_index(raw_data.minutes)
        
        # 计算价格趋势
        price_trend = self._compute_price_momentum(
            [raw_data.price],  # 这里需要历史价格数据
            self.price_window
        )
        
        return {
            'form_trajectory': form_trajectory,
            'fatigue_index': fatigue,
            'price_trend': price_trend
        }
        
    def _compute_form_trajectory(self, 
                               points: Union[List[int], int], 
                               minutes: Union[List[int], int]) -> np.ndarray:
        """计算状态轨迹"""
        # 处理单个值的情况
        if isinstance(points, (int, float)):
            points = [points]
        if isinstance(minutes, (int, float)):
            minutes = [minutes]
            
        if not points:
            return np.zeros(self.form_window)
            
        # 计算每分钟得分
        points_per_min = [
            p/m if m > 0 else 0 
            for p, m in zip(points, minutes)
        ]
        
        # 使用指数加权移动平均
        weights = np.exp(np.linspace(-1, 0, self.form_window))
        weights /= weights.sum()
        
        # 如果数据点不够，复制最后一个值
        while len(points_per_min) < self.form_window:
            points_per_min.append(points_per_min[-1] if points_per_min else 0)
        
        trajectory = np.convolve(
            points_per_min[-self.form_window:],
            weights,
            mode='valid'
        )
        
        return np.pad(
            trajectory,
            (0, max(0, self.form_window - len(trajectory))),
            mode='edge'
        )
        
    def _compute_fatigue_index(self, minutes: Union[List[int], int]) -> float:
        """计算疲劳指数"""
        # 处理单个值的情况
        if isinstance(minutes, (int, float)):
            minutes = [minutes]
            
        if not minutes:
            return 0.0
            
        recent_minutes = minutes[-3:]  # 考虑最近3轮
        # 如果数据点不够，复制最后一个值
        while len(recent_minutes) < 3:
            recent_minutes.append(recent_minutes[-1] if recent_minutes else 0)
            
        weights = [0.5, 0.3, 0.2]  # 赋予不同权重
        
        fatigue = sum(
            m/90 * w 
            for m, w in zip(recent_minutes, weights)
        )
        
        return min(1.0, fatigue)
        
    def _compute_price_momentum(self, 
                              prices: Union[List[float], float], 
                              window: int) -> np.ndarray:
        """计算价格动量"""
        # 处理单个值的情况
        if isinstance(prices, (int, float)):
            prices = [prices]
            
        if len(prices) < 2:
            # 如果只有一个价格，假设没有变化
            return np.zeros(window)
            
        # 如果数据点不够，复制最后一个值
        while len(prices) < window + 1:
            prices.append(prices[-1])
            
        changes = np.diff(prices)
        padded_changes = np.pad(
            changes,
            (0, max(0, window - len(changes))),
            mode='edge'
        )
        
        return padded_changes[-window:]
        
    def create_temporal_state(self,
                            players_data: Dict[int, Dict],
                            fixture_difficulty: Dict[int, List[Dict]],
                            timestamp: datetime) -> TemporalState:
        """创建时间维度状态"""
        return TemporalState(
            timestamp=timestamp,
            fixture_difficulty={
                pid: self._process_fixture_difficulty(fixtures)
                for pid, fixtures in fixture_difficulty.items()
            },
            fatigue_index={
                pid: data['fatigue_index']
                for pid, data in players_data.items()
            },
            form_trajectory={
                pid: data['form_trajectory']
                for pid, data in players_data.items()
            },
            price_trend={
                pid: data['price_trend']
                for pid, data in players_data.items()
            }
        )
        
    def _process_fixture_difficulty(self, fixtures: Union[List[Dict], float, List[float], int]) -> float:
        """处理赛程难度
        
        Args:
            fixtures: 可以是以下类型之一：
                - List[Dict]: 包含difficulty键的字典列表
                - float: 直接的难度值
                - List[float]: 难度值列表
                - int: 整数难度值
        
        Returns:
            float: 标准化的难度值（0-1之间）
        """
        # 如果是单个数值，直接返回标准化值
        if isinstance(fixtures, (float, int)):
            return float(fixtures) / 5.0
        
        # 如果是空列表，返回默认难度
        if not fixtures:
            return 1.0
        
        # 如果是数值列表
        if isinstance(fixtures[0], (float, int)):
            difficulties = fixtures[:3]  # 考虑未来3轮
        # 如果是字典列表
        else:
            difficulties = [f.get('difficulty', 3) for f in fixtures[:3]]  # 考虑未来3轮
        
        # 如果数据点不够3个，用最后一个值填充
        while len(difficulties) < 3:
            difficulties.append(difficulties[-1] if difficulties else 3)
        
        weights = [0.5, 0.3, 0.2]  # 权重递减
        
        # 计算加权平均难度并标准化到0-1
        return sum(d * w for d, w in zip(difficulties, weights)) / 5.0 