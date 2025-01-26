import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableChecker(nn.Module):
    """可微分约束检查器"""
    def __init__(self, budget_limit=100.0, max_players=600):
        super().__init__()
        self.budget_limit = budget_limit
        self.max_players = max_players
        
        # 约束权重学习器
        self.constraint_weights = nn.Parameter(torch.ones(4))
        
        # 约束违反惩罚因子
        self.penalty_factors = {
            'budget': 2.0,
            'position': 1.5,
            'team': 1.0,
            'value': 0.5
        }
    
    def _pad_tensor(self, tensor, target_size):
        """将张量填充到目标大小（处理1D、2D和3D张量）"""
        # 确保张量至少是2维的
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # [num_players] -> [1, num_players]
        
        if tensor.dim() == 2:
            # 如果是独热编码，保持特征维度不变
            if tensor.size(-1) in [4, 20]:  # 位置(4)或球队(20)独热编码
                # [num_players, features] -> [1, num_players, features]
                tensor = tensor.unsqueeze(0)
                padded = torch.zeros(1, target_size, tensor.size(-1), device=tensor.device)
                padded[0, :tensor.size(1), :] = tensor
                return padded
            else:
                # 其他2D张量的填充
                if tensor.size(1) < target_size:
                    padding = (0, target_size - tensor.size(1))
                    return F.pad(tensor, padding, mode='constant', value=0)
                return tensor[:, :target_size]
        
        elif tensor.dim() == 3:
            if tensor.size(1) < target_size:
                padded = torch.zeros(tensor.size(0), target_size, tensor.size(2), device=tensor.device)
                padded[:, :tensor.size(1), :] = tensor
                return padded
            return tensor[:, :target_size, :]
        
        return tensor
    
    def check_budget_constraint(self, selected_values):
        """检查预算约束"""
        total_cost = selected_values.sum(dim=1)  # [batch_size]
        violation = F.relu(total_cost - self.budget_limit)
        return violation * self.penalty_factors['budget']
    
    def check_position_constraint(self, selected_positions, required_counts):
        """检查位置约束"""
        # [batch_size, num_players, 4] -> [batch_size, 4]
        position_counts = selected_positions.sum(dim=1)
        violations = F.relu(torch.abs(position_counts - required_counts))
        return violations.mean(dim=1) * self.penalty_factors['position']
    
    def check_team_constraint(self, selected_teams, max_per_team=3):
        """检查球队约束"""
        # [batch_size, num_players, 20] -> [batch_size, 20]
        team_counts = selected_teams.sum(dim=1)
        violations = F.relu(team_counts - max_per_team)
        return violations.mean(dim=1) * self.penalty_factors['team']
    
    def check_value_constraint(self, selected_values, selected_points):
        """检查价值效率约束"""
        # [batch_size]
        value_efficiency = selected_points.sum(dim=1) / (selected_values.sum(dim=1) + 1e-6)
        violation = F.relu(0.5 - value_efficiency)
        return violation * self.penalty_factors['value']
    
    def forward(self, state):
        """
        检查约束违反情况
        Args:
            state: 包含以下键的字典：
                - selections: [num_players, 1] 选择的球员
                - values: [num_players] 球员价值
                - position_onehot: [num_players, num_positions] 位置独热编码
                - team_onehot: [num_players, num_teams] 球队独热编码
                - player_features: [num_players, feature_dim] 球员特征
        """
        selections = state['selections']
        values = state['values']
        position_onehot = state['position_onehot']
        team_onehot = state['team_onehot']
        
        # 1. 预算约束
        total_value = torch.sum(selections * values.unsqueeze(-1))
        budget_violation = F.relu(total_value - 100.0)
        
        # 2. 位置约束
        position_counts = torch.sum(selections * position_onehot, dim=0)
        required_counts = torch.tensor([2, 5, 5, 3], device=selections.device)  # GK, DEF, MID, FWD
        position_violation = torch.sum(torch.abs(position_counts - required_counts))
        
        # 3. 球队约束
        team_counts = torch.sum(selections * team_onehot, dim=0)
        team_violation = F.relu(torch.max(team_counts) - 3.0)
        
        # 4. 总球员数约束
        total_players = torch.sum(selections)
        player_count_violation = torch.abs(total_players - 15.0)
        
        return {
            'budget': budget_violation,
            'position': position_violation,
            'team': team_violation,
            'total_players': player_count_violation,
            'total': budget_violation + position_violation + team_violation + player_count_violation
        } 