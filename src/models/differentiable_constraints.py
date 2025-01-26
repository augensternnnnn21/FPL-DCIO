import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union
from .base_constraints import BaseConstraintChecker

class DifferentiableChecker(BaseConstraintChecker, nn.Module):
    """可微分约束检查器，用于神经网络的约束检查"""
    def __init__(self, budget_limit=100.0, max_players=15):
        BaseConstraintChecker.__init__(self)
        nn.Module.__init__(self)
        
        self.budget_limit = budget_limit
        self.max_players = max_players
        
        # 约束权重学习器
        self.constraint_weights = nn.Parameter(torch.ones(4))
        
        # 约束违反惩罚因子
        self.penalty_factors = {
            'budget': 2.0,
            'position': 1.5,
            'team': 1.0,
            'total': 1.0
        }
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return self.check_constraints(state)
    
    def check_constraints(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """检查约束违反情况"""
        device = state['selections'].device
        selections = state['selections']
        values = state['values']
        position_onehot = state['position_onehot']
        team_onehot = state['team_onehot']
        
        # 1. 预算约束
        total_value = torch.sum(selections * values)
        budget_violation = F.relu(total_value - self.budget_limit)
        
        # 2. 位置约束
        position_counts = torch.sum(selections.unsqueeze(-1) * position_onehot, dim=0)
        required_counts = torch.tensor([2, 5, 5, 3], device=device)
        position_violation = torch.sum(torch.abs(position_counts - required_counts))
        
        # 3. 球队约束
        team_counts = torch.sum(selections.unsqueeze(-1) * team_onehot, dim=0)
        team_violation = F.relu(torch.max(team_counts) - self.max_per_team)
        
        # 4. 总球员数约束
        total_players = torch.sum(selections)
        player_count_violation = torch.abs(total_players - self.max_players)
        
        # 应用惩罚因子
        violations = {
            'budget': budget_violation * self.penalty_factors['budget'],
            'position': position_violation * self.penalty_factors['position'],
            'team': team_violation * self.penalty_factors['team'],
            'total_players': player_count_violation * self.penalty_factors['total']
        }
        
        # 计算总违规
        violations['total'] = (
            self.constraint_weights[0] * violations['budget'] +
            self.constraint_weights[1] * violations['position'] +
            self.constraint_weights[2] * violations['team'] +
            self.constraint_weights[3] * violations['total_players']
        )
        
        return violations
    
    def get_reward(self, violations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算奖励（添加缩放）"""
        return -violations['total'] / 100.0