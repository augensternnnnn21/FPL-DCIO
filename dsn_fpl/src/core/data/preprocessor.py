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
        self.feature_means = {}
        self.feature_stds = {}
        self.position_weights = {
            'GK': 1.8,   # 进一步提高门将权重
            'DEF': 1.5,  # 进一步提高后卫权重
            'MID': 1.3,  # 提高中场权重
            'FWD': 1.2   # 略微提高前锋权重
        }
        self.feature_importance = {
            'total_points': 1.0,
            'form': 0.9,
            'points_per_game': 0.8,
            'minutes': 0.7,
            'selected_by_percent': 0.6
        }
        
    def standardize_features(self, data: Dict[str, float]) -> Dict[str, float]:
        """使用改进的特征标准化方法
        
        Args:
            data: 原始特征数据
            
        Returns:
            Dict[str, float]: 标准化后的特征数据
        """
        standardized = {}
        numerical_features = []
        
        # 第一步：数据清洗和异常值处理
        for key, value in data.items():
            if key in ['id', 'name', 'team', 'position']:
                standardized[key] = value
                continue
                
            try:
                value = float(value)
                # 处理异常值
                if key in ['minutes', 'total_points', 'form', 'points_per_game']:
                    value = np.clip(value, 0, np.inf)  # 不允许负值
                if key == 'selected_by_percent':
                    value = np.clip(value, 0, 100)  # 百分比限制
                numerical_features.append((key, value))
            except (ValueError, TypeError):
                standardized[key] = value
                continue
        
        # 第二步：分层标准化
        for key, value in numerical_features:
            if key not in self.feature_means:
                self.feature_means[key] = value
                self.feature_stds[key] = 1.0
            
            # 使用Winsorization进行异常值处理
            p1, p99 = np.percentile([value], [1, 99])
            value = np.clip(value, p1, p99)
            
            # 使用改进的标准化方法
            if key in ['total_points', 'form', 'points_per_game']:
                # 对关键指标使用Min-Max标准化
                min_val = 0
                max_val = max(100, self.feature_means[key] * 1.5)  # 更保守的最大值估计
                std_value = (value - min_val) / (max_val - min_val + 1e-8)
                # 使用更保守的压缩方法
                std_value = np.clip(std_value, 0, 1)
            else:
                # 其他指标使用Robust标准化
                median = self.feature_means[key]
                iqr = self.feature_stds[key]
                if iqr == 0:
                    iqr = 1.0
                std_value = (value - median) / (iqr + 1e-8)
                # 使用更温和的压缩
                std_value = 0.5 * (1 + np.tanh(std_value / 2))
            
            # 应用特征重要性权重，但使用更小的权重
            if key in self.feature_importance:
                std_value *= (0.5 + 0.5 * self.feature_importance[key])
            
            # 确保最终值在合理范围内
            standardized[key] = float(np.clip(std_value, 0, 1))
            
        return standardized
        
    def process_player_data(self, player_data: RawPlayerData) -> Dict:
        """处理球员数据，添加更多特征组合
        
        Args:
            player_data: 原始球员数据
            
        Returns:
            Dict: 处理后的球员数据
        """
        # 基础特征提取
        processed = {
            'id': player_data.player_id,
            'name': player_data.name,
            'team': player_data.team,
            'position': player_data.position,
            'minutes': float(player_data.minutes or 0),
            'form': float(player_data.form or 0),
            'points_per_game': float(player_data.points_per_game or 0),
            'selected_by_percent': float(player_data.selected_by_percent or 0),
            'total_points': float(player_data.total_points or 0)
        }
        
        # 高级特征工程
        minutes = float(player_data.minutes or 0)
        form = float(player_data.form or 0)
        points_per_game = float(player_data.points_per_game or 0)
        total_points = float(player_data.total_points or 0)
        selected_by = float(player_data.selected_by_percent or 0)
        
        # 1. 效率指标
        processed['points_per_90'] = (points_per_game * 90) / (minutes + 1e-8)
        processed['points_per_selection'] = total_points / (selected_by + 1e-8)
        
        # 2. 稳定性指标
        processed['performance_stability'] = 1 - np.abs(form - points_per_game) / (points_per_game + 1e-8)
        processed['consistency_score'] = np.exp(-np.abs(form - points_per_game))
        
        # 3. 趋势指标
        processed['form_momentum'] = form * np.log1p(minutes / 90)
        processed['popularity_momentum'] = form * np.log1p(selected_by)
        
        # 4. 综合评分
        processed['overall_rating'] = (
            processed['points_per_90'] * 0.3 +
            processed['performance_stability'] * 0.2 +
            processed['form_momentum'] * 0.3 +
            processed['popularity_momentum'] * 0.2
        )
        
        # 5. 位置特定特征
        if processed['position'] == 'GK':
            processed['save_potential'] = points_per_game * 1.5
            processed['clean_sheet_value'] = total_points / (minutes/90 + 1e-8) * 1.3
        elif processed['position'] == 'DEF':
            processed['clean_sheet_potential'] = points_per_game * 1.3
            processed['attacking_threat'] = form * 1.2
        elif processed['position'] == 'MID':
            processed['involvement_potential'] = points_per_game * 1.2
            processed['creativity_score'] = form * 1.4
        elif processed['position'] == 'FWD':
            processed['goal_potential'] = points_per_game * 1.4
            processed['scoring_efficiency'] = total_points / (minutes + 1e-8) * 90
        
        # 标准化特征
        processed = self.standardize_features(processed)
        
        # 位置权重调整
        position_weight = self.position_weights.get(player_data.position, 1.0)
        
        # 动态权重调整
        if minutes > 60:  # 主力球员
            position_weight *= 1.2
        if form > points_per_game:  # 状态上升期
            position_weight *= 1.15
        if selected_by > 20:  # 热门球员
            position_weight *= 1.1
            
        # 应用权重
        for key in processed:
            if isinstance(processed[key], (int, float)) and key not in ['id']:
                processed[key] *= position_weight
        
        return processed
        
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
        
    def create_temporal_state(self, processed_data: Dict, fixture_difficulty: Dict, current_time: datetime) -> TemporalState:
        """创建时间维度状态
        
        Args:
            processed_data: 处理后的数据
            fixture_difficulty: 赛程难度
            current_time: 当前时间
            
        Returns:
            TemporalState: 时间维度状态
        """
        # 计算疲劳指数
        fatigue_indices = {}
        for pid, data in processed_data.items():
            minutes = float(data.get('minutes', 0))
            fatigue = 1.0 - np.exp(-minutes / 9000)  # 使用指数衰减
            fatigue_indices[pid] = max(0.0, min(1.0, fatigue))  # 限制在[0,1]范围内
            
        # 计算状态轨迹
        form_trajectories = {}
        for pid, data in processed_data.items():
            form = float(data.get('form', 0))
            points_per_game = float(data.get('points_per_game', 0))
            trajectory = np.array([form, points_per_game])
            form_trajectories[pid] = trajectory
            
        # 计算价格趋势
        price_trends = {}
        for pid, data in processed_data.items():
            selected_by = float(data.get('selected_by_percent', 0))
            price_trend = np.tanh(selected_by / 100)  # 使用双曲正切函数归一化
            price_trends[pid] = price_trend
            
        return TemporalState(
            timestamp=current_time,
            fixture_difficulty=fixture_difficulty,
            fatigue_index=fatigue_indices,
            form_trajectory=form_trajectories,
            price_trend=price_trends
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