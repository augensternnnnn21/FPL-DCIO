from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class RippleType(Enum):
    INJURY = "injury"
    WEATHER = "weather"
    SOCIAL = "social"
    TACTICAL = "tactical"
    
@dataclass
class RippleEvent:
    """扰动事件"""
    type: RippleType
    intensity: float  # 0-1
    source_id: str
    affected_ids: List[str]
    decay_rate: float
    
class RippleWave:
    """扰动波"""
    def __init__(self, event: RippleEvent):
        self.event = event
        self.propagation_matrix = np.zeros((10, 10))  # 假设有10个维度的传播空间
        self._initialize_wave()
        
    def _initialize_wave(self):
        """初始化扰动波的传播矩阵"""
        intensity = self.event.intensity
        for i in range(10):
            for j in range(10):
                distance = abs(i - j)
                self.propagation_matrix[i][j] = intensity * np.exp(-distance * self.event.decay_rate)

    def propagate(self, time_step: int) -> np.ndarray:
        """计算扰动波在特定时间步的传播效果"""
        return self.propagation_matrix * np.exp(-time_step * self.event.decay_rate)

class ChaosRippleGenerator:
    def __init__(self):
        self.active_ripples: List[RippleWave] = []
        self.influence_cache: Dict[str, np.ndarray] = {}
        
    def add_ripple(self, event: RippleEvent):
        """添加新的扰动事件"""
        wave = RippleWave(event)
        self.active_ripples.append(wave)
        self._update_influence_cache(event.affected_ids)
        
    def _update_influence_cache(self, affected_ids: List[str]):
        """更新受影响实体的影响缓存"""
        for entity_id in affected_ids:
            total_influence = np.zeros((10,))
            for ripple in self.active_ripples:
                total_influence += ripple.propagate(0).sum(axis=0)
            self.influence_cache[entity_id] = total_influence
            
    def get_entity_perturbation(self, entity_id: str) -> np.ndarray:
        """获取特定实体受到的扰动影响"""
        return self.influence_cache.get(entity_id, np.zeros((10,)))
    
    def simulate_butterfly_effect(self, 
                                initial_event: RippleEvent, 
                                steps: int) -> List[Tuple[str, float]]:
        """模拟蝴蝶效应的连锁反应"""
        impacts = []
        current_wave = RippleWave(initial_event)
        
        for step in range(steps):
            propagation = current_wave.propagate(step)
            for entity_id in initial_event.affected_ids:
                impact = float(propagation.sum())
                impacts.append((entity_id, impact))
                
            # 如果影响超过阈值，可能触发新的扰动事件
            if max(impact for _, impact in impacts[-len(initial_event.affected_ids):]) > 0.7:
                secondary_event = RippleEvent(
                    type=initial_event.type,
                    intensity=initial_event.intensity * 0.8,
                    source_id=initial_event.affected_ids[0],
                    affected_ids=initial_event.affected_ids[1:],
                    decay_rate=initial_event.decay_rate * 1.2
                )
                self.add_ripple(secondary_event)
                
        return impacts 