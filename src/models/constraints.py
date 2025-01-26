import pandas as pd
from typing import Dict, Union
from .base_constraints import BaseConstraintChecker

class ConstraintChecker(BaseConstraintChecker):
    """基本约束检查器，用于非神经网络的约束检查"""
    
    def check_constraints(self, 
                        selected_players: pd.DataFrame,
                        budget: float,
                        max_players: int) -> float:
        """检查所有约束并返回奖励"""
        reward = 0.0
        
        # 1. 检查位置约束
        position_penalty = self._check_position_constraints(selected_players)
        reward -= position_penalty
        
        # 2. 检查球队约束
        team_penalty = self._check_team_constraints(selected_players)
        reward -= team_penalty
        
        # 3. 检查预算约束
        budget_penalty = self._check_budget_constraints(selected_players, budget)
        reward -= budget_penalty
        
        # 4. 检查总人数约束
        count_penalty = self._check_count_constraints(selected_players, max_players)
        reward -= count_penalty
        
        # 5. 添加得分奖励
        points_reward = selected_players['total_points'].sum()
        reward += points_reward
        
        return reward
    
    def get_reward(self, violations: Dict[str, float]) -> float:
        return -sum(violations.values())
    
    def _check_position_constraints(self, players: pd.DataFrame) -> float:
        """检查位置约束"""
        penalty = 0.0
        position_counts = players['position'].value_counts()
        
        for pos, required in self.position_limits.items():
            actual = position_counts.get(pos, 0)
            if actual != required:
                penalty += 100 * abs(actual - required)
        
        return penalty
    
    def _check_team_constraints(self, players: pd.DataFrame) -> float:
        """检查球队约束"""
        team_counts = players['team'].value_counts()
        if team_counts.max() > self.max_per_team:
            return 100 * (team_counts.max() - self.max_per_team)
        return 0.0
    
    def _check_budget_constraints(self, players: pd.DataFrame, budget: float) -> float:
        """检查预算约束"""
        total_value = players['value'].sum()
        if total_value > budget:
            return 100 * (total_value - budget)
        return 0.0
    
    def _check_count_constraints(self, players: pd.DataFrame, max_players: int) -> float:
        """检查总人数约束"""
        if len(players) != max_players:
            return 100 * abs(len(players) - max_players)
        return 0.0