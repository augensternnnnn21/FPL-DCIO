import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class QuantumState:
    """球员的量子状态"""
    current_form: np.ndarray  # 当前状态向量
    
    def __init__(self):
        # 初始化为12维的量子状态向量
        self.current_form = np.random.rand(12)
        # 归一化
        self.current_form = self.current_form / np.linalg.norm(self.current_form)
        
    def update(self, new_stats: Dict):
        """更新量子状态
        
        Args:
            new_stats: 新的统计数据
        """
        # 根据新的统计数据更新状态向量
        form_factor = float(new_stats.get('form', 0)) / 10.0
        momentum = float(new_stats.get('momentum', 0)) / 10.0
        potential = float(new_stats.get('potential', 0.5))
        
        # 应用量子演化
        evolution_matrix = np.eye(12) + 0.1 * np.random.randn(12, 12)
        evolution_matrix = evolution_matrix / np.linalg.norm(evolution_matrix, axis=1, keepdims=True)
        
        # 更新状态
        self.current_form = evolution_matrix @ self.current_form
        
        # 添加统计信息的影响
        influence = np.array([
            form_factor,
            momentum,
            potential,
            float(new_stats.get('minutes', 0)) / 90.0,
            float(new_stats.get('goals_scored', 0)) / 3.0,
            float(new_stats.get('assists', 0)) / 3.0,
            float(new_stats.get('clean_sheets', 0)) / 1.0,
            float(new_stats.get('saves', 0)) / 5.0,
            float(new_stats.get('bonus', 0)) / 3.0,
            float(new_stats.get('bps', 0)) / 1000.0,
            float(new_stats.get('influence', 0)) / 100.0,
            float(new_stats.get('creativity', 0)) / 100.0
        ])
        
        # 混合新的影响
        self.current_form = 0.7 * self.current_form + 0.3 * influence
        
        # 重新归一化
        self.current_form = self.current_form / np.linalg.norm(self.current_form)

class QPlayer:
    """量子球员类"""
    def __init__(
        self,
        player_id: int,
        name: str,
        position: str,
        base_stats: Optional[Dict] = None
    ):
        """初始化量子球员
        
        Args:
            player_id: 球员ID
            name: 球员名字
            position: 位置
            base_stats: 基础统计数据
        """
        self.player_id = player_id
        self.name = name
        self.position = position
        self._quantum_state = QuantumState()
        
        if base_stats:
            self._quantum_state.update(base_stats)
            
    def update_state(self, new_stats: Dict):
        """更新球员状态
        
        Args:
            new_stats: 新的统计数据
        """
        self._quantum_state.update(new_stats)

    def compute_synergy(self, other_player: 'QPlayer') -> float:
        """计算与其他球员的量子协同效应"""
        state_product = np.dot(
            self._quantum_state.current_form,
            other_player._quantum_state.current_form
        )
        synergy_factor = np.trace(
            np.dot(
                self._quantum_state.synergy_matrix,
                other_player._quantum_state.synergy_matrix
            )
        )
        return state_product * synergy_factor * (
            self._quantum_state.potential_burst * 
            other_player._quantum_state.potential_burst
        ) 