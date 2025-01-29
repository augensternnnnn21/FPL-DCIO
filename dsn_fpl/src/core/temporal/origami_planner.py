from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

class TimeScale(Enum):
    IMMEDIATE = "immediate"  # 当前轮次
    SHORT_TERM = "short_term"  # 2-3轮
    MEDIUM_TERM = "medium_term"  # 4-6轮
    LONG_TERM = "long_term"  # 7轮以上

@dataclass
class TemporalState:
    """时间维度状态"""
    timestamp: datetime
    fixture_difficulty: Dict[int, float]  # 赛程难度
    fatigue_index: Dict[int, float]  # 疲劳指数
    form_trajectory: Dict[int, List[float]]  # 状态轨迹
    price_trend: Dict[int, List[float]]  # 价格趋势

@dataclass
class TimeSlice:
    """时间切片"""
    scale: TimeScale
    weight: float
    states: List[TemporalState]
    
class TemporalOrigamiPlanner:
    def __init__(self):
        self.time_slices: Dict[TimeScale, TimeSlice] = {}
        self.folded_states: Dict[int, np.ndarray] = {}  # 球员ID -> 折叠后的状态向量
        self.decision_weights = {
            TimeScale.IMMEDIATE: 0.4,
            TimeScale.SHORT_TERM: 0.3,
            TimeScale.MEDIUM_TERM: 0.2,
            TimeScale.LONG_TERM: 0.1
        }
        
    def add_temporal_state(self, state: TemporalState):
        """添加时间状态"""
        for scale in TimeScale:
            if scale not in self.time_slices:
                self.time_slices[scale] = TimeSlice(
                    scale=scale,
                    weight=self.decision_weights[scale],
                    states=[]
                )
            self.time_slices[scale].states.append(state)
        self._fold_time_dimensions()
        
    def _fold_time_dimensions(self):
        """执行时间维度折叠"""
        for player_id in self._get_all_player_ids():
            folded_state = np.zeros(4)  # 4个时间尺度
            
            for i, scale in enumerate(TimeScale):
                time_slice = self.time_slices[scale]
                scale_state = self._compute_scale_state(player_id, time_slice)
                folded_state[i] = scale_state * time_slice.weight
                
            self.folded_states[player_id] = folded_state
            
    def _compute_scale_state(self, player_id: int, time_slice: TimeSlice) -> float:
        """计算特定时间尺度下的状态"""
        states = time_slice.states
        if not states:
            return 0.0
            
        # 综合考虑多个因素
        difficulty_score = np.mean([s.fixture_difficulty.get(player_id, 0) for s in states])
        fatigue_score = np.mean([s.fatigue_index.get(player_id, 0) for s in states])
        form_score = np.mean([s.form_trajectory.get(player_id, [0])[-1] for s in states])
        price_trend = np.mean([s.price_trend.get(player_id, [0])[-1] for s in states])
        
        return (form_score * 0.4 + 
                (1 - difficulty_score) * 0.3 + 
                (1 - fatigue_score) * 0.2 + 
                price_trend * 0.1)
    
    def _get_all_player_ids(self) -> Set[int]:
        """获取所有球员ID"""
        player_ids = set()
        for time_slice in self.time_slices.values():
            for state in time_slice.states:
                player_ids.update(state.fixture_difficulty.keys())
        return player_ids
    
    def get_optimal_timing(self, player_id: int) -> Tuple[TimeScale, float]:
        """获取最优时机"""
        if player_id not in self.folded_states:
            return TimeScale.IMMEDIATE, 0.0
            
        folded_state = self.folded_states[player_id]
        best_scale_idx = np.argmax(folded_state)
        return list(TimeScale)[best_scale_idx], folded_state[best_scale_idx]
    
    def generate_transfer_plan(self, 
                             current_team: List[int],
                             budget: float,
                             num_transfers: int) -> List[Tuple[int, int, TimeScale]]:
        """生成转会计划"""
        plans = []
        potential_targets = self._get_all_player_ids() - set(current_team)
        
        # 对当前阵容进行时间维度评估
        team_evaluations = {
            pid: self.folded_states.get(pid, np.zeros(4)).sum()
            for pid in current_team
        }
        
        # 找出需要替换的球员
        to_replace = sorted(
            team_evaluations.items(),
            key=lambda x: x[1]
        )[:num_transfers]
        
        # 为每个待替换球员找到最佳替代者
        for old_id, _ in to_replace:
            best_target = None
            best_score = float('-inf')
            best_timing = TimeScale.IMMEDIATE
            
            for target_id in potential_targets:
                if target_id not in self.folded_states:
                    continue
                    
                timing, score = self.get_optimal_timing(target_id)
                if score > best_score:
                    best_target = target_id
                    best_score = score
                    best_timing = timing
                    
            if best_target:
                plans.append((old_id, best_target, best_timing))
                potential_targets.remove(best_target)
                
        return plans 