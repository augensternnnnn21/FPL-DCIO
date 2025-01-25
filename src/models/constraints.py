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
        """将张量填充到目标大小（处理2D和3D张量）"""
        if tensor.dim() == 2:  # [batch_size, num_players]
            if tensor.size(1) < target_size:
                padding = (0, target_size - tensor.size(1))  # 在第二个维度填充
                return F.pad(tensor, padding, mode='constant', value=0)
            else:
                return tensor[:, :target_size]
        elif tensor.dim() == 3:  # [batch_size, num_players, features]
            if tensor.size(1) < target_size:
                padding = (0, 0, 0, target_size - tensor.size(1))  # 在第二个维度填充
                return F.pad(tensor, padding, mode='constant', value=0)
            else:
                return tensor[:, :target_size, :]
        return tensor
    
    def check_budget_constraint(self, selected_values):
        """检查预算约束"""
        total_cost = selected_values.sum(dim=1)
        violation = F.relu(total_cost - self.budget_limit)
        return violation * self.penalty_factors['budget']
    
    def check_position_constraint(self, selected_positions, required_counts):
        """检查位置约束"""
        position_counts = selected_positions.sum(dim=1)  # [batch_size, num_positions]
        violations = F.relu(torch.abs(position_counts - required_counts))
        return violations.mean(dim=1) * self.penalty_factors['position']
    
    def check_team_constraint(self, selected_teams, max_per_team=3):
        """检查球队约束"""
        team_counts = selected_teams.sum(dim=1)  # [batch_size, num_teams]
        violations = F.relu(team_counts - max_per_team)
        return violations.mean(dim=1) * self.penalty_factors['team']
    
    def check_value_constraint(self, selected_values, selected_points):
        """检查价值效率约束"""
        value_efficiency = selected_points.sum(dim=1) / (selected_values.sum(dim=1) + 1e-6)
        violation = F.relu(0.5 - value_efficiency)  # 要求至少0.5的价值效率
        return violation * self.penalty_factors['value']
    
    def forward(self, selections, player_data):
        """
        前向传播检查所有约束
        Args:
            selections: [batch_size, num_players] 选择概率
            player_data: 字典，包含以下键：
                - 'values': [batch_size, num_players] 球员价值
                - 'position_onehot': [batch_size, num_players, 4] 位置独热编码
                - 'team_onehot': [batch_size, num_players, 20] 球队独热编码
                - 'points': [batch_size, num_players] 球员得分
        """
        batch_size = selections.size(0)
        device = selections.device
        
        # 确保 selections 维度正确
        selections = self._pad_tensor(selections, self.max_players)  # [batch_size, 600]
        
        # 确保 player_data 中的张量也填充到 max_players
        padded_player_data = {}
        for key in player_data:
            if key in ['position_onehot', 'team_onehot']:
                # 处理三维张量 [batch_size, num_players, features]
                padded_player_data[key] = self._pad_tensor(player_data[key], self.max_players)
            else:
                # 处理二维张量 [batch_size, num_players]
                padded_player_data[key] = self._pad_tensor(player_data[key], self.max_players)
        
        violations = {}
        
        # 1. 预算约束
        if 'values' in player_data:
            values = padded_player_data['values']
            assert selections.size(1) == values.size(1), \
                f"维度不匹配：selections {selections.shape}, values {values.shape}"
            violations['budget'] = self.check_budget_constraint(selections * values)
        else:
            violations['budget'] = torch.tensor(0.0, device=device)
        
        # 2. 位置约束
        if 'position_onehot' in player_data:
            position_onehot = padded_player_data['position_onehot']
            assert selections.size(1) == position_onehot.size(1), \
                f"维度不匹配：selections {selections.shape}, position_onehot {position_onehot.shape}"
            violations['position'] = self.check_position_constraint(
                selections.unsqueeze(-1) * position_onehot,
                torch.tensor([2, 5, 5, 3], device=device)
            )
        else:
            violations['position'] = torch.tensor(0.0, device=device)
        
        # 3. 球队约束
        if 'team_onehot' in player_data:
            team_onehot = padded_player_data['team_onehot']
            assert selections.size(1) == team_onehot.size(1), \
                f"维度不匹配：selections {selections.shape}, team_onehot {team_onehot.shape}"
            violations['team'] = self.check_team_constraint(
                selections.unsqueeze(-1) * team_onehot
            )
        else:
            violations['team'] = torch.tensor(0.0, device=device)
        
        # 4. 价值约束
        if 'values' in player_data and 'points' in player_data:
            values = padded_player_data['values']
            points = padded_player_data['points']
            assert selections.size(1) == values.size(1) == points.size(1), \
                f"维度不匹配：selections {selections.shape}, values {values.shape}, points {points.shape}"
            violations['value'] = self.check_value_constraint(
                selections * values,
                selections * points
            )
        else:
            violations['value'] = torch.tensor(0.0, device=device)
        
        # 加权组合所有违反
        total_violation = sum(
            w * v for w, v in zip(F.softmax(self.constraint_weights, dim=0), 
                                violations.values())
        )
        
        return {
            'total': total_violation,
            'details': violations
        } 