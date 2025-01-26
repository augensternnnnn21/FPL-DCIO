import mip
import torch
import numpy as np
import logging
from typing import Dict, Optional

class NeuralILPSolver:
    def __init__(self, budget=100.0, max_per_team=3, total_players=15):
        self.budget = budget
        self.max_per_team = max_per_team
        self.total_players = total_players
        self.timeout = 30  # 秒
        self.max_retries = 3
        self.logger = logging.getLogger("FPL-DCIO")

    def solve(self, 
             action_probs: torch.Tensor, 
             values: torch.Tensor, 
             teams: np.ndarray,
             positions: Optional[np.ndarray] = None,
             current_squad: Optional[np.ndarray] = None) -> Dict:
        """
        增强版ILP求解器
        
        Args:
            action_probs: 选择概率
            values: 球员价值
            teams: 球队信息
            positions: 位置信息 (可选)
            current_squad: 当前阵容 (可选,用于限制转会次数)
        
        Returns:
            Dict: {
                'selected': 选择的球员数组,
                'status': 求解状态,
                'objective_value': 目标函数值
            }
        """
        for attempt in range(self.max_retries):
            try:
                model = self._build_model(
                    action_probs, values, teams, positions, current_squad
                )
                
                # 求解
                status = model.optimize(max_seconds=self.timeout)
                
                if status == mip.OptimizationStatus.OPTIMAL:
                    selected = self._extract_solution(model)
                    return {
                        'selected': selected,
                        'status': 'optimal',
                        'objective_value': model.objective_value
                    }
                    
                elif status == mip.OptimizationStatus.FEASIBLE:
                    self.logger.warning("Found feasible but not optimal solution")
                    selected = self._extract_solution(model)
                    return {
                        'selected': selected,
                        'status': 'feasible',
                        'objective_value': model.objective_value
                    }
                    
            except Exception as e:
                self.logger.error(f"Solver error (attempt {attempt+1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return self._fallback_solution(action_probs, values, teams)
                continue
        
        return self._fallback_solution(action_probs, values, teams)

    def _build_model(self, action_probs, values, teams, positions=None, current_squad=None):
        """构建ILP模型"""
        n = len(action_probs)
        model = mip.Model()
        
        # 决策变量
        x = [model.add_var(var_type=mip.BINARY) for _ in range(n)]
        
        # 目标函数
        model.objective = mip.maximize(
            mip.xsum(float(action_probs[i].item()) * x[i] for i in range(n))
        )
        
        # 约束条件
        self._add_constraints(model, x, values, teams, positions, current_squad)
        
        return model

    def _add_constraints(self, model, x, values, teams, positions, current_squad):
        """添加约束条件"""
        n = len(x)
        
        # 1. 预算约束
        model.add_constr(
            mip.xsum(float(values[i].item()) * x[i] for i in range(n)) <= self.budget
        )
        
        # 2. 总人数约束
        model.add_constr(
            mip.xsum(x[i] for i in range(n)) == self.total_players
        )
        
        # 3. 每队人数约束
        for team in np.unique(teams):
            team_indices = [i for i in range(n) if teams[i] == team]
            if team_indices:
                model.add_constr(
                    mip.xsum(x[i] for i in team_indices) <= self.max_per_team
                )
        
        # 4. 位置约束
        if positions is not None:
            pos_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            for pos, required in pos_requirements.items():
                pos_indices = [i for i in range(n) if positions[i] == pos]
                if pos_indices:
                    model.add_constr(
                        mip.xsum(x[i] for i in pos_indices) == required
                    )
        
        # 5. 转会限制(如果提供当前阵容)
        if current_squad is not None:
            max_transfers = 2  # 可配置
            changes = mip.xsum(
                abs(x[i] - current_squad[i]) for i in range(n)
            ) / 2  # 除2因为每次转会计算两次变化
            model.add_constr(changes <= max_transfers)

    def _extract_solution(self, model):
        """从模型中提取解决方案"""
        return np.array([1 if x.x >= 0.99 else 0 for x in model.vars])

    def _fallback_solution(self, action_probs, values, teams):
        """后备方案：使用贪心算法"""
        self.logger.warning("Using fallback greedy solution")
        return {
            'selected': self._greedy_selection(action_probs, values, teams),
            'status': 'fallback',
            'objective_value': None
        }

    def _greedy_selection(self, action_probs, values, teams, positions=None):
        """贪心选择算法作为后备方案"""
        n = len(action_probs)
        selected = np.zeros(n)
        
        # 将球员按照性价比排序 (得分概率/价值)
        value_ratio = action_probs.cpu().numpy() / values.cpu().numpy()
        
        # 如果提供了位置信息，按位置选择
        if positions is not None:
            pos_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            
            for pos, required in pos_requirements.items():
                # 获取该位置的球员
                pos_mask = positions == pos
                pos_indices = np.where(pos_mask)[0]
                
                if len(pos_indices) < required:
                    self.logger.warning(f"Not enough {pos} players available")
                    continue
                
                # 按性价比排序
                pos_ratios = value_ratio[pos_indices]
                sorted_indices = pos_indices[np.argsort(-pos_ratios)]
                
                # 选择前required个球员，同时考虑预算和球队限制
                selected_count = 0
                for idx in sorted_indices:
                    if selected_count >= required:
                        break
                        
                    # 检查预算
                    current_value = (selected * values.cpu().numpy()).sum()
                    if current_value + values[idx].item() > self.budget:
                        continue
                    
                    # 检查球队限制
                    team = teams[idx]
                    team_count = sum(1 for i in range(n) if selected[i] == 1 and teams[i] == team)
                    if team_count >= self.max_per_team:
                        continue
                    
                    selected[idx] = 1
                    selected_count += 1
        
        else:
            # 如果没有位置信息，简单选择前15个满足约束的球员
            sorted_indices = np.argsort(-value_ratio)
            selected_count = 0
            
            for idx in sorted_indices:
                if selected_count >= self.total_players:
                    break
                    
                # 检查预算
                current_value = (selected * values.cpu().numpy()).sum()
                if current_value + values[idx].item() > self.budget:
                    continue
                
                # 检查球队限制
                team = teams[idx]
                team_count = sum(1 for i in range(n) if selected[i] == 1 and teams[i] == team)
                if team_count >= self.max_per_team:
                    continue
                
                selected[idx] = 1
                selected_count += 1
        
        return selected 