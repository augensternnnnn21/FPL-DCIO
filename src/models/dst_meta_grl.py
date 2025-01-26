import torch
import torch.nn as nn
import torch.nn.functional as F
from .relation_net import RelationNet
from .st_conv import STConv
from .meta_learning import MetaDistiller, CurriculumScheduler, GradientAggregator
from .constraints import DifferentiableChecker
import numpy as np

class DynamicSpatioTemporalMetaLearner(nn.Module):
    """动态时空元图强化学习模型"""
    def __init__(self, input_dim=12, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1. 动态异构图构建层
        self.relation_learner = RelationNet(input_dim=input_dim)
        
        # 2. 时空融合模块
        self.st_conv = STConv(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gcn_types=['position', 'team', 'value'],
            temporal_win=5
        )
        
        # 3. 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 从input_dim投影到hidden_dim
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4. 元学习组件
        self.curriculum = CurriculumScheduler()
        self.grad_aggregator = GradientAggregator()
        self.meta_distiller = MetaDistiller(
            inner_loop=self.curriculum,
            outer_loop=self.grad_aggregator
        )
        
        # 5. 约束处理器
        self.constraint_checker = DifferentiableChecker()
        
        # 6. 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # 7. 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        
        # 8. 动作生成器
        self.action_generator = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 600)  # 假设最大球员数为600
        )
        
        # 修改这部分，延迟导入泛化组件
        from .generalization import DomainAdversarialTraining, SeasonalAdaptationNetwork
        self.domain_adversarial = DomainAdversarialTraining(feature_dim=hidden_dim)
        self.seasonal_adaptation = SeasonalAdaptationNetwork(hidden_dim)
        
        # 添加赛季编码器
        self.season_encoder = nn.Sequential(
            nn.Linear(4, 16),  # 4个季节特征：年份、是否COVID赛季等
            nn.ReLU(),
            nn.LayerNorm(16)
        )
    
    def get_representation(self, state):
        """获取状态表示，用于泛化性评估"""
        with torch.no_grad():
            output = self.forward(state)
            return output['domain_features']
    
    def forward(self, state):
        """前向传播"""
        batch_size = 1
        if isinstance(state, dict):
            features = state['player_features']
        else:
            # 如果state是张量，直接使用
            features = state
            if not isinstance(features, torch.Tensor):
                features = torch.FloatTensor(features)
            if features.dim() == 2:
                features = features.unsqueeze(0)  # 添加batch维度
        
        # 打印特征维度
        print("\nFeature dimensions:")
        print(f"Input features shape: {features.shape}")
        
        # 1. 特征编码
        encoded_features = self.feature_proj(features)
        print(f"Encoded features shape: {encoded_features.shape}")
        
        # 2. 时空信息融合
        spatial_info = torch.cat([
            features[..., -24:],  # 假设最后24维是空间信息
            encoded_features
        ], dim=-1)
        print(f"Spatial info shape: {spatial_info.shape}")
        
        # 3. 时空卷积
        print("\nSTConv input shapes:")
        print(f"x: {encoded_features.shape}")
        print(f"spatial_info: {spatial_info.shape}")
        
        conv_out = self.st_conv(
            x=encoded_features,
            spatial_info=spatial_info
        )
        print(f"After TCN shape: {conv_out.shape}")
        
        # 4. 多头注意力
        position_onehot = features[..., :4]  # 假设前4维是位置的one-hot编码
        team_onehot = features[..., 4:24]  # 假设4-24维是球队的one-hot编码
        values = features[..., -1]  # 假设最后一维是价值
        
        print("\nConstraint state:")
        print(f"values shape: {values.shape}")
        print(f"position_onehot shape: {position_onehot.shape}")
        print(f"team_onehot shape: {team_onehot.shape}")
        print(f"player_features shape: {features.shape}")
        
        # 5. 计算约束违反
        constraint_state = {
            'selections': torch.ones(features.size(1), 1).to(features.device),  # [num_players, 1]
            'values': values.squeeze(0),  # [num_players]
            'position_onehot': position_onehot.squeeze(0),  # [num_players, num_positions]
            'team_onehot': team_onehot.squeeze(0),  # [num_players, num_teams]
            'player_features': features.squeeze(0)  # [num_players, feature_dim]
        }
        
        violations = self.constraint_checker(constraint_state)
        
        # 6. 输出动作概率和价值
        action_probs = F.softmax(self.policy_net(conv_out), dim=-1)
        state_value = self.value_net(conv_out)
        
        return {
            'action_probs': action_probs,
            'state_value': state_value,
            'violations': violations
        }
    
    def compute_loss(self, episode_data, season_info=None):
        """计算损失"""
        # 从 episode_data 中提取数据
        states = [data['state'] for data in episode_data]
        actions = [data['action'] for data in episode_data]
        rewards = [data['reward'] for data in episode_data]
        next_states = [data['next_state'] for data in episode_data]
        dones = [data['done'] for data in episode_data]
        
        # 转换为张量
        device = next(self.parameters()).device
        rewards = torch.tensor(rewards, device=device).float()
        dones = torch.tensor(dones, device=device).float()
        
        total_policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_value_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 遍历每个时间步
        for t in range(len(episode_data)):
            # 获取当前步的数据
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            next_state = next_states[t]
            done = dones[t]
            
            # 前向传播
            output = self.forward(state)
            action_probs = output['action_probs']
            state_value = output['state_value']
            
            # 计算策略损失
            if action_probs.dim() == 3:
                action_probs = action_probs[0]  # [num_players, max_players]
            
            # 确保动作是长整型
            if isinstance(action, torch.Tensor):
                action = action.long()
            else:
                action = torch.tensor(action, device=device).long()
            
            policy_loss = F.cross_entropy(action_probs, action)
            
            # 计算价值损失
            if isinstance(reward, (int, float)):
                target_value = torch.tensor([[reward]], device=device).float()
            else:
                target_value = reward.view(-1, 1)
            
            # 调整状态值的维度以匹配目标值
            state_value = state_value.view_as(target_value)
            value_loss = F.mse_loss(state_value, target_value)
            
            # 累加损失
            total_policy_loss = total_policy_loss + policy_loss
            total_value_loss = total_value_loss + value_loss
        
        # 计算平均损失
        batch_size = len(episode_data)
        avg_policy_loss = total_policy_loss / batch_size
        avg_value_loss = total_value_loss / batch_size
        
        # 计算总损失
        total_loss = avg_policy_loss + 0.5 * avg_value_loss
        
        return {
            'policy_loss': avg_policy_loss,  # 返回张量而不是标量
            'value_loss': avg_value_loss,    # 返回张量而不是标量
            'total_loss': total_loss         # 返回张量而不是标量
        }
    
    def select_action(self, state):
        """选择动作"""
        with torch.no_grad():
            output = self.forward(state)
            action_probs = output['action_probs']
            
            # 获取位置和价值信息，并确保是PyTorch张量
            if isinstance(state, dict):
                position_onehot = state['position_onehot']
                values = state['values']
            else:
                position_onehot = state[..., :4]
                values = state[..., -1]
            
            # 转换为PyTorch张量
            if isinstance(position_onehot, np.ndarray):
                position_onehot = torch.from_numpy(position_onehot).float()
            if isinstance(values, np.ndarray):
                values = torch.from_numpy(values).float()
            
            # 确保维度正确
            if action_probs.dim() == 3:
                action_probs = action_probs[0]
            if position_onehot.dim() == 3:
                position_onehot = position_onehot[0]
            if isinstance(values, torch.Tensor) and values.dim() == 2:
                values = values[0]
            
            print(f"\nDimension check:")
            print(f"position_onehot shape: {position_onehot.shape}")
            print(f"values shape: {values.shape}")
            print(f"action_probs shape: {action_probs.shape}")
            
            # 初始化选择
            num_players = action_probs.size(0)
            device = action_probs.device
            action = torch.zeros(num_players, dtype=torch.bool, device=device)
            
            # 阵容要求
            required_counts = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            position_indices = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
            
            print("\nPosition selection:")
            positions_priority = ['GK', 'DEF', 'MID', 'FWD']
            selected_by_position = {}
            total_value = 0
            
            # 第一步：确保每个位置至少有最低要求的人数
            for pos in positions_priority:
                pos_idx = position_indices[pos]
                # 修改这里：直接使用2维索引
                pos_mask = position_onehot[:, pos_idx] == 1
                required = required_counts[pos]
                
                if not pos_mask.any():
                    print(f"Warning: No {pos} players available!")
                    return None
                
                # 获取该位置所有球员的价值和概率
                pos_values = values[pos_mask]
                pos_probs = action_probs[pos_mask].mean(dim=1)
                
                # 计算性价比分数
                value_per_point = pos_probs / (pos_values + 1e-6)
                
                # 选择所需数量的球员
                num_available = pos_mask.sum().item()
                num_to_select = min(required, num_available)
                
                if num_to_select < required:
                    print(f"Warning: Not enough {pos} players! Need {required}, found {num_available}")
                    return None
                
                # 根据性价比选择球员
                _, indices = torch.topk(value_per_point, k=num_to_select)
                pos_players = torch.where(pos_mask)[0][indices]
                
                # 更新选择
                action[pos_players] = True
                selected_by_position[pos] = num_to_select
                total_value += pos_values[indices].sum().item()
                
                print(f"{pos}: found {num_available} players, selected {num_to_select}")
            
            # 检查预算约束
            if total_value > 100:
                print(f"Warning: Total value ({total_value:.1f}) exceeds budget!")
                return None
            
            # 检查是否满足最低要求
            total_selected = action.sum().item()
            if total_selected != 15:
                print(f"Warning: Could only select {total_selected} players")
                return None
            
            # 打印最终选择结果
            print("\nFinal selection:")
            print(f"Total players: {total_selected}")
            print(f"Total value: {total_value:.1f}")
            for pos, count in selected_by_position.items():
                print(f"{pos}: {count} players")
            
            return {
                'action': action,
                'log_prob': torch.log(action_probs + 1e-10).mean(dim=1),
                'value': output['state_value'],
                'violations': output['violations']
            }
    
    def _build_position_adj_matrix(self, position_onehot):
        """构建基于位置的邻接矩阵"""
        # [batch_size, num_players, num_players]
        return torch.matmul(position_onehot, position_onehot.transpose(-2, -1))
    
    def _build_team_adj_matrix(self, team_onehot):
        """构建基于球队的邻接矩阵"""
        # [batch_size, num_players, num_players]
        return torch.matmul(team_onehot, team_onehot.transpose(-2, -1))
    
    def _build_value_adj_matrix(self, values):
        """构建基于价值的邻接矩阵"""
        # [batch_size, num_players, 1]
        values_expanded = values.unsqueeze(-1)
        # 计算价值差异
        value_diff = torch.abs(values_expanded - values_expanded.transpose(-2, -1))
        # 使用高斯核将差异转换为相似度
        sigma = values.std()
        return torch.exp(-value_diff / (2 * sigma * sigma)) 