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
    
    def forward(self, selections, player_data):
        """前向传播检查所有约束"""
        # 首先修正 selections 的维度
        if selections.dim() == 3:
            selections = selections.squeeze(0)  # [1, 511, 600] -> [511, 600]
        batch_size = selections.size(0)
        device = selections.device
        
        # 打印输入维度
        print(f"\nInput shapes:")
        print(f"selections: {selections.shape}")
        for key, value in player_data.items():
            if value is not None:  # 只打印非None值的形状
                print(f"{key}: {value.shape}")
        
        # 处理 player_data
        padded_player_data = {}
        for key, value in player_data.items():
            if value is None:  # 跳过None值
                continue
            
            if key in ['position_onehot', 'team_onehot']:
                # 处理独热编码张量
                if value.dim() == 2:  # [num_players, features]
                    # 确定特征数量
                    num_features = 4 if key == 'position_onehot' else 20
                    # 确保特征维度正确
                    if value.size(-1) != num_features:
                        value = value[:, :num_features]
                    # [num_players, features] -> [1, num_players, features]
                    value = value.unsqueeze(0)
                
                # 填充到目标大小
                padded = torch.zeros(1, self.max_players, value.size(-1), device=device)
                padded[0, :value.size(1), :] = value[0, :value.size(1), :]
                # 扩展到 batch_size
                padded = padded.expand(batch_size, -1, -1)
                padded_player_data[key] = padded
            
            elif key == 'history':
                # 处理历史数据 [time_steps, num_players, features]
                if value.dim() == 3:
                    time_steps, num_players, features = value.shape
                    padded = torch.zeros(time_steps, self.max_players, features, device=device)
                    padded[:, :num_players, :] = value
                    padded_player_data[key] = padded
            
            else:
                # 处理其他张量 (features, values, points等)
                if value.dim() == 1:  # [num_players]
                    value = value.unsqueeze(0)  # -> [1, num_players]
                elif value.dim() == 2:  # [num_players, features]
                    # 保持特征维度不变，只填充球员维度
                    padded = torch.zeros(batch_size, self.max_players, value.size(-1), device=device)
                    padded[:, :value.size(0), :] = value.unsqueeze(0).expand(batch_size, -1, -1)
                    padded_player_data[key] = padded
                    continue
                
                # 填充到目标大小 (对于1D张量)
                padded = torch.zeros(batch_size, self.max_players, device=device)
                padded[:, :value.size(1)] = value.expand(batch_size, -1)
                padded_player_data[key] = padded
        
        # 打印填充后的维度
        print(f"\nPadded shapes:")
        print(f"selections: {selections.shape}")
        for key, value in padded_player_data.items():
            print(f"{key}: {value.shape}")
        
        violations = {}
        
        # 1. 预算约束
        if 'values' in padded_player_data:  # 使用padded_player_data而不是player_data
            values = padded_player_data['values']
            if values.dim() == 3:  # [batch_size, num_players, 1]
                values = values.squeeze(-1)
            violations['budget'] = self.check_budget_constraint(selections * values)
            print(f"Budget violation shape: {violations['budget'].shape}")
        
        # 2. 位置约束
        if 'position_onehot' in padded_player_data:
            position_onehot = padded_player_data['position_onehot']
            position_probs = selections.unsqueeze(-1) * position_onehot
            violations['position'] = self.check_position_constraint(
                position_probs,
                torch.tensor([2, 5, 5, 3], device=device)
            )
            print(f"Position violation shape: {violations['position'].shape}")
        
        # 3. 球队约束
        if 'team_onehot' in padded_player_data:
            team_onehot = padded_player_data['team_onehot']
            team_probs = selections.unsqueeze(-1) * team_onehot
            violations['team'] = self.check_team_constraint(team_probs)
            print(f"Team violation shape: {violations['team'].shape}")
        
        # 4. 价值约束
        if all(k in padded_player_data for k in ['values', 'points']):
            values = padded_player_data['values']
            points = padded_player_data['points']
            violations['value'] = self.check_value_constraint(
                selections * values,
                selections * points
            )
            print(f"Value violation shape: {violations['value'].shape}")
        
        # 确保所有违反项均为一维张量 [batch_size]
        for key, value in violations.items():
            assert value.dim() == 1, f"{key} violation has wrong dimension: {value.shape}"
        
        # 加权组合所有违反
        weights = F.softmax(self.constraint_weights, dim=0)
        total_violation = sum(w * v for w, v in zip(weights, violations.values()))
        
        return {
            'total': total_violation,
            'details': violations
        } 