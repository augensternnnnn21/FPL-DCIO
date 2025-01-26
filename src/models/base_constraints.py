from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import pandas as pd
import torch

class BaseConstraintChecker(ABC):
    """约束检查器基类"""
    def __init__(self):
        self.position_limits = {
            'GK': 2,
            'DEF': 5,
            'MID': 5,
            'FWD': 3
        }
        self.max_per_team = 3
        self.total_players = 15
    
    @abstractmethod
    def check_constraints(self, state: Any) -> Dict[str, Union[float, torch.Tensor]]:
        """检查约束并返回违规程度"""
        pass
    
    @abstractmethod
    def get_reward(self, violations: Dict[str, Union[float, torch.Tensor]]) -> Union[float, torch.Tensor]:
        """根据违规计算奖励"""
        pass 