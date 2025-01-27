import pulp
import numpy as np
import logging
from typing import Dict, Optional

class NeuralILPSolver:
    def __init__(self, budget=100.0, max_per_team=3, total_players=15):
        self.budget = budget
        self.max_per_team = max_per_team
        self.total_players = total_players
        self.timeout = 30
        self.max_retries = 3
        self.logger = logging.getLogger("FPL-DCIO")

    def solve(self, 
             action_probs: np.ndarray, 
             values: np.ndarray, 
             teams: np.ndarray) -> np.ndarray:
        """使用 PuLP 求解器"""
        try:
            # 创建问题
            prob = pulp.LpProblem("FPL_Selection", pulp.LpMaximize)
            
            # 决策变量
            n = len(values)
            x = pulp.LpVariable.dicts("player", 
                                    range(n),
                                    cat='Binary')
            
            # 目标函数
            prob += pulp.lpSum([action_probs[i] * x[i] for i in range(n)])
            
            # 约束条件
            # 1. 预算约束
            prob += pulp.lpSum([values[i] * x[i] for i in range(n)]) <= self.budget
            
            # 2. 总人数约束
            prob += pulp.lpSum([x[i] for i in range(n)]) == self.total_players
            
            # 3. 每支球队最多3人约束
            if len(teams) > 0:
                unique_teams = np.unique(teams)
                for team in unique_teams:
                    team_indices = np.where(teams == team)[0]
                    if len(team_indices) > 0:
                        prob += pulp.lpSum([x[i] for i in team_indices]) <= self.max_per_team
            
            # 求解
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            
            # 获取结果
            selected = np.zeros(n)
            if pulp.LpStatus[prob.status] == 'Optimal':
                for i in range(n):
                    selected[i] = pulp.value(x[i])
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Solver error: {str(e)}")
            # 使用贪心算法作为后备方案
            return self._greedy_fallback(action_probs, values, teams)
    
    def _greedy_fallback(self, action_probs, values, teams):
        """贪心算法作为后备方案"""
        n = len(values)
        selected = np.zeros(n)
        value_ratio = action_probs / (values + 1e-6)  # 避免除零
        
        # 按价值比排序
        sorted_indices = np.argsort(-value_ratio)
        selected_count = 0
        total_cost = 0
        team_counts = {}
        
        for idx in sorted_indices:
            if selected_count >= self.total_players:
                break
                
            # 检查预算
            if total_cost + values[idx] > self.budget:
                continue
            
            # 检查球队限制
            if len(teams) > 0:
                team = teams[idx]
                if team_counts.get(team, 0) >= self.max_per_team:
                    continue
                team_counts[team] = team_counts.get(team, 0) + 1
            
            selected[idx] = 1
            selected_count += 1
            total_cost += values[idx]
        
        return selected 