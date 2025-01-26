import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class DilatedTCN(nn.Module):
    """带有扩张的时间卷积网络"""
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 n_layers: int = 3,  # 使用 n_layers 而不是 num_layers
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 bidirectional: bool = True,
                 device: str = None):  # 添加 device 参数
        super().__init__()
        
        # 设置设备
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.gamma = 0.99  # 添加折扣因子
        
        # 修改输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 时序卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=2**i)
            for i in range(n_layers)
        ])
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # 输出层
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_layer = nn.Linear(self.output_dim, hidden_dim)
        
        # 修改动作头和值函数头的输入维度
        self.feature_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 添加损失函数
        self.criterion = nn.MSELoss()
        
        # 将模型移动到指定设备
        self.to(self.device)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据，可能的形状:
               - [seq_len, feature_dim] 或 [feature_dim, seq_len]
               - [batch_size, seq_len, feature_dim]
        """
        # 打印输入形状以便调试
        print(f"Input shape: {x.shape}")
        
        # 确保输入是3D张量
        if len(x.shape) == 2:
            # 确保第二个维度是特征维度
            if x.shape[1] != self.input_dim and x.shape[0] != self.input_dim:
                # 如果两个维度都不匹配，则需要调整特征维度
                x = x[:, :self.input_dim] if x.shape[1] > self.input_dim else F.pad(x, (0, self.input_dim - x.shape[1]))
            
            # 如果第一个维度是特征维度，则转置
            if x.shape[0] == self.input_dim:
                x = x.t()
            
            x = x.unsqueeze(0)  # 添加批次维度 [1, seq_len, feature_dim]
        
        # 如果是3D张量，确保最后一个维度是特征维度
        else:
            if x.shape[-1] != self.input_dim:
                x = x[..., :self.input_dim] if x.shape[-1] > self.input_dim else F.pad(x, (0, self.input_dim - x.shape[-1]))
        
        # 再次打印形状以确认转换结果
        print(f"After reshape: {x.shape}")
        
        # 输入投影
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # 转换为卷积格式
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        
        # 应用卷积层
        for conv in self.conv_layers:
            padding = conv.dilation[0] * (conv.kernel_size[0] - 1)
            x = F.pad(x, (padding, 0))  # 添加填充以保持序列长度
            x = conv(x)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # 转回序列格式
        x = x.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
        
        # 应用注意力机制
        if self.use_attention:
            x, _ = self.attention(x, x, x)
        
        # 如果是双向，连接正向和反向特征
        if self.bidirectional:
            forward = x
            backward = torch.flip(x, [1])
            x = torch.cat([forward, backward], dim=-1)
        
        # 输出层
        x = self.output_layer(x)  # [batch_size, seq_len, hidden_dim]
        
        return x

    def select_action(self, state):
        """选择动作"""
        try:
            # 获取特征
            features = self.forward(state['player_features'])  # [batch_size, seq_len, hidden_dim]
            
            # 全局池化
            features = features.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
            features = self.feature_pooling(features)  # [batch_size, hidden_dim, 1]
            features = features.squeeze(-1)  # [batch_size, hidden_dim]
            
            # 获取位置信息
            position_onehot = state['position_onehot']  # [seq_len, 4]
            values = state['raw_features']['values']
            num_players = position_onehot.size(0)
            
            # 确保 position_onehot 是正确的形状
            if len(position_onehot.shape) == 2:
                position_onehot = position_onehot.unsqueeze(0)  # [1, seq_len, 4]
            
            # 为每个球员生成动作概率
            action_probs = []
            for i in range(num_players):
                player_features = features.clone()  # [batch_size, hidden_dim]
                # 生成该球员的动作概率
                prob = torch.sigmoid(self.action_head(player_features))  # [batch_size, 1]
                action_probs.append(prob)
            
            # 组合所有球员的概率
            action_probs = torch.cat(action_probs, dim=1)  # [batch_size, num_players]
            
            # 创建掩码来满足约束
            batch_size = position_onehot.size(0)
            
            # 初始化动作
            action = torch.zeros_like(action_probs)
            
            # 获取每个位置的球员索引
            gk_mask = position_onehot[..., 0] == 1  # GK
            def_mask = position_onehot[..., 1] == 1  # DEF
            mid_mask = position_onehot[..., 2] == 1  # MID
            fwd_mask = position_onehot[..., 3] == 1  # FWD
            
            # 打印调试信息
            print(f"\nDEBUG: Shapes:")
            print(f"action_probs: {action_probs.shape}")
            print(f"position_onehot: {position_onehot.shape}")
            print(f"gk_mask: {gk_mask.shape}")
            
            print(f"\nDEBUG: Position counts:")
            print(f"GK count: {gk_mask.sum().item()}")
            print(f"DEF count: {def_mask.sum().item()}")
            print(f"MID count: {mid_mask.sum().item()}")
            print(f"FWD count: {fwd_mask.sum().item()}")
            
            # 按照概率排序选择球员
            for b in range(batch_size):
                # 选择2名门将
                gk_probs = action_probs[b][gk_mask[b]]
                if len(gk_probs) >= 2:  # 确保有足够的门将可选
                    gk_indices = torch.argsort(gk_probs, descending=True)[:2]
                    gk_positions = torch.where(gk_mask[b])[0]
                    action[b][gk_positions[gk_indices]] = 1
                
                # 选择5名后卫
                def_probs = action_probs[b][def_mask[b]]
                if len(def_probs) >= 5:  # 确保有足够的后卫可选
                    def_indices = torch.argsort(def_probs, descending=True)[:5]
                    def_positions = torch.where(def_mask[b])[0]
                    action[b][def_positions[def_indices]] = 1
                
                # 选择5名中场
                mid_probs = action_probs[b][mid_mask[b]]
                if len(mid_probs) >= 5:  # 确保有足够的中场可选
                    mid_indices = torch.argsort(mid_probs, descending=True)[:5]
                    mid_positions = torch.where(mid_mask[b])[0]
                    action[b][mid_positions[mid_indices]] = 1
                
                # 选择3名前锋
                fwd_probs = action_probs[b][fwd_mask[b]]
                if len(fwd_probs) >= 3:  # 确保有足够的前锋可选
                    fwd_indices = torch.argsort(fwd_probs, descending=True)[:3]
                    fwd_positions = torch.where(fwd_mask[b])[0]
                    action[b][fwd_positions[fwd_indices]] = 1
            
            # 打印选择结果
            print(f"\nDEBUG: Selection results:")
            print(f"Total selected: {action.sum().item()}")
            print(f"Selected by position:")
            for b in range(batch_size):
                print(f"Batch {b}:")
                print(f"GK: {(action[b] * gk_mask[b]).sum().item()}")
                print(f"DEF: {(action[b] * def_mask[b]).sum().item()}")
                print(f"MID: {(action[b] * mid_mask[b]).sum().item()}")
                print(f"FWD: {(action[b] * fwd_mask[b]).sum().item()}")
            
            return action.detach()
            
        except Exception as e:
            print(f"Error in select_action: {str(e)}")
            print(f"State shape before forward: {state['player_features'].shape}")
            print(f"Position onehot shape: {state['position_onehot'].shape}")
            print(f"Action probs shape: {action_probs.shape if 'action_probs' in locals() else 'Not created'}")
            raise
    
    def get_representation(self, state):
        """获取状态表示"""
        if isinstance(state, dict):
            state = state['player_features']
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
            
        return self.forward(state)
    
    def compute_loss(self, episode_data, season):
        """计算损失"""
        try:
            # 转换状态为张量
            states = episode_data['states']
            actions = torch.FloatTensor(episode_data['actions']).to(self.device)
            rewards = torch.FloatTensor(episode_data['rewards']).to(self.device)
            next_states = episode_data['next_states']
            dones = torch.FloatTensor(episode_data['dones']).to(self.device)
            
            # 处理状态和下一个状态
            # 使用填充来处理不同长度的序列
            max_len = max(s['player_features'].shape[0] for s in states)
            
            def pad_features(features, max_len):
                """填充特征到最大长度"""
                pad_size = max_len - features.shape[0]
                if pad_size > 0:
                    # 创建填充
                    padding = np.zeros((pad_size,) + features.shape[1:])
                    # 连接原始特征和填充
                    return np.concatenate([features, padding], axis=0)
                return features
            
            # 填充并堆叠特征
            padded_states = np.stack([
                pad_features(s['player_features'], max_len) for s in states
            ])
            padded_next_states = np.stack([
                pad_features(s['player_features'], max_len) for s in next_states
            ])
            
            # 转换为张量
            state_features = torch.FloatTensor(padded_states).to(self.device)
            next_state_features = torch.FloatTensor(padded_next_states).to(self.device)
            
            # 创建掩码来忽略填充的部分
            state_lengths = torch.tensor([
                s['player_features'].shape[0] for s in states
            ]).to(self.device)
            
            # 计算当前Q值
            current_q = self.forward(state_features)  # [batch_size, max_len, hidden_dim]
            
            # 计算目标Q值
            with torch.no_grad():
                next_q = self.forward(next_state_features)  # [batch_size, max_len, hidden_dim]
                
                # 创建掩码
                mask = torch.arange(max_len).unsqueeze(0).to(self.device) < state_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).float()  # [batch_size, max_len, 1]
                
                # 应用掩码
                next_q = next_q * mask
                
                # 对每个序列进行池化
                next_q = next_q.sum(dim=1) / state_lengths.unsqueeze(1)  # [batch_size, hidden_dim]
                
                # 计算目标值
                target_q = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * self.gamma * next_q
            
            # 对当前Q值也应用相同的掩码和池化
            current_q = current_q * mask
            current_q = current_q.sum(dim=1) / state_lengths.unsqueeze(1)  # [batch_size, hidden_dim]
            
            # 计算策略损失和值函数损失
            policy_loss = self.criterion(current_q, target_q)
            value_loss = F.mse_loss(self.value_head(current_q).squeeze(), rewards)
            
            # 添加正则化损失
            l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.parameters())
            
            # 计算总损失
            total_loss = policy_loss + 0.5 * value_loss + l2_reg
            
            # 添加调试信息
            if total_loss.item() > 100:
                print(f"\nDEBUG: Loss breakdown:")
                print(f"Policy loss: {policy_loss.item()}")
                print(f"Value loss: {value_loss.item()}")
                print(f"L2 reg: {l2_reg.item()}")
                print(f"Total loss: {total_loss.item()}")
                print(f"Average reward: {rewards.mean().item()}")
            
            return {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'total_loss': total_loss,
                'avg_reward': rewards.mean().item()
            }
            
        except Exception as e:
            print(f"Error in compute_loss: {str(e)}")
            print(f"States shapes: {[s['player_features'].shape for s in states]}")
            print(f"Actions shape: {actions.shape}")
            print(f"Rewards shape: {rewards.shape}")
            print(f"Max length: {max_len}")
            print(f"State lengths: {[s['player_features'].shape[0] for s in states]}")
            raise

class GATFusion(nn.Module):
    """图注意力融合模块"""
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 输入投影
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)  # 修改这里，因为输入已经是 hidden_dim
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x, adj_matrix):
        """
        前向传播
        Args:
            x: [batch_size, num_players, hidden_dim]  # 注意：输入已经是 hidden_dim 维度
            adj_matrix: [batch_size, num_players, num_players]
        """
        batch_size = x.size(0)
        num_players = x.size(1)
        
        # 投影输入
        x = self.input_proj(x)  # [batch_size, num_players, hidden_dim]
        
        # 调整掩码维度以匹配多头注意力的要求
        attn_mask = (adj_matrix == 0)
        if len(attn_mask.shape) == 3:
            # 扩展掩码维度以匹配注意力头数
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_mask = attn_mask.reshape(batch_size * self.num_heads, attn_mask.size(-2), attn_mask.size(-1))
        
        # 应用注意力机制
        out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        
        # 残差连接和输出投影
        out = x + self.output_proj(out)  # [batch_size, num_players, hidden_dim]
        
        return out

class TemporalConvNet(nn.Module):
    """时间卷积网络"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.tcn = nn.Sequential(
            # 第一层：从input_dim到hidden_dim
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            # 第二层：保持hidden_dim
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, num_players, input_dim]
        Returns:
            输出张量 [batch_size, num_players, hidden_dim]
        """
        # 打印输入维度
        print(f"TCN input shape before permute: {x.shape}")
        
        # 调整维度顺序：[batch_size, num_players, input_dim] -> [batch_size, input_dim, num_players]
        x = x.permute(0, 2, 1)
        print(f"TCN input shape after permute: {x.shape}")
        
        # 应用TCN
        x = self.tcn(x)
        print(f"TCN output shape before permute back: {x.shape}")
        
        # 恢复维度顺序：[batch_size, hidden_dim, num_players] -> [batch_size, num_players, hidden_dim]
        x = x.permute(0, 2, 1)
        print(f"TCN final output shape: {x.shape}")
        
        return x

class STConv(nn.Module):
    """时空卷积模块"""
    def __init__(self, input_dim, hidden_dim=64, gcn_types=None, temporal_win=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temporal_win = temporal_win
        
        # 时间卷积
        self.tcn = TemporalConvNet(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
        # 空间图卷积
        if gcn_types is None:
            gcn_types = ['position', 'team', 'value']
        self.gcn_types = gcn_types
        
        self.gcns = nn.ModuleDict({
            gtype: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for gtype in gcn_types
        })
        
        # 图注意力融合
        self.gat_fusion = GATFusion(hidden_dim, hidden_dim)
    
    def _build_position_adj_matrix(self, position_onehot):
        """
        构建基于位置的邻接矩阵
        Args:
            position_onehot: [batch_size, num_players, num_positions]
        Returns:
            邻接矩阵 [batch_size, num_players, num_players]
        """
        # 打印输入维度
        print(f"Position one-hot shape: {position_onehot.shape}")
        
        # 计算相似度矩阵
        adj = torch.matmul(position_onehot, position_onehot.transpose(-2, -1))
        
        # 归一化
        adj = F.softmax(adj, dim=-1)
        
        print(f"Position adjacency matrix shape: {adj.shape}")
        return adj
    
    def _build_team_adj_matrix(self, team_onehot):
        """
        构建基于球队的邻接矩阵
        Args:
            team_onehot: [batch_size, num_players, num_teams]
        Returns:
            邻接矩阵 [batch_size, num_players, num_players]
        """
        # 打印输入维度
        print(f"Team one-hot shape: {team_onehot.shape}")
        
        # 计算相似度矩阵
        adj = torch.matmul(team_onehot, team_onehot.transpose(-2, -1))
        
        # 归一化
        adj = F.softmax(adj, dim=-1)
        
        print(f"Team adjacency matrix shape: {adj.shape}")
        return adj
    
    def _build_value_adj_matrix(self, values):
        """
        构建基于价值的邻接矩阵
        Args:
            values: [batch_size, num_players]
        Returns:
            邻接矩阵 [batch_size, num_players, num_players]
        """
        # 打印输入维度
        print(f"Value feature shape: {values.shape}")
        
        # 扩展维度以便计算差异
        values = values.unsqueeze(-1)  # [batch_size, num_players, 1]
        
        # 计算价值差异
        diff = torch.abs(values - values.transpose(-2, -1))  # [batch_size, num_players, num_players]
        
        # 使用高斯核将差异转换为相似度
        sigma = values.std()
        adj = torch.exp(-diff / (2 * sigma * sigma))
        
        # 归一化
        adj = F.softmax(adj, dim=-1)
        
        print(f"Value adjacency matrix shape: {adj.shape}")
        return adj
    
    def forward(self, x, spatial_info):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, num_players, hidden_dim]
            spatial_info: 空间信息 [batch_size, num_players, spatial_dim]
        """
        # 打印输入维度
        print(f"STConv input shapes:")
        print(f"x: {x.shape}")
        print(f"spatial_info: {spatial_info.shape}")
        
        # 1. 时间卷积
        x = self.tcn(x)
        print(f"After TCN shape: {x.shape}")
        
        # 2. 空间图卷积
        for gtype in self.gcn_types:
            # 构建邻接矩阵
            if gtype == 'position':
                adj = self._build_position_adj_matrix(spatial_info[..., :4])
            elif gtype == 'team':
                adj = self._build_team_adj_matrix(spatial_info[..., 4:])
            else:  # value
                adj = self._build_value_adj_matrix(x.mean(dim=-1))
            
            # 应用图注意力融合
            x = self.gat_fusion(x, adj)
            print(f"After {gtype} GAT fusion shape: {x.shape}")
        
        return x 