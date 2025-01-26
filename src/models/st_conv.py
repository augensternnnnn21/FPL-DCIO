import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from src.models.neural_ilp import NeuralILPSolver

class DilatedTCN(nn.Module):
    """带有扩张的时间卷积网络"""
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 n_layers: int = 3,  # 使用 n_layers 而不是 num_layers
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 bidirectional: bool = True,
                 device: str = None,
                 budget: float = 100.0,
                 use_ilp: bool = False,
                 use_synergy: bool = False):  # 添加预算参数
        super().__init__()
        
        # 设置设备和预算
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.budget = budget  # 添加预算属性
        self.use_ilp = use_ilp
        self.use_synergy = use_synergy
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.gamma = 0.99  # 添加折扣因子
        
        # 修改输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)  # 这里保留LayerNorm因为输入是[batch_size, seq_len, hidden_dim]
        )
        
        # 修改卷积层结构，使用GroupNorm
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, 
                             dilation=2**i, padding=2**i),
                    nn.GroupNorm(8, hidden_dim),  # 使用8个组进行归一化
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
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
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 限制价值输出范围
        )
        
        # 添加损失函数
        self.criterion = nn.MSELoss()
        
        # 将模型移动到指定设备
        self.to(self.device)
        
        # 添加经验回放相关属性
        self.replay_buffer = []
        self.min_replay_size = 1000
        self.max_buffer_size = 10000
        
        self.gat_fusion = GATFusion(hidden_dim, hidden_dim)  # 或按照实际维度设置
        
        self.ilp_solver = NeuralILPSolver(budget=budget, max_per_team=3, total_players=15)
        
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
        
        # 转置为卷积格式
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        
        # 应用卷积层
        for conv in self.conv_layers:
            x = conv(x)  # 维持 [batch_size, hidden_dim, seq_len]
        
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

    def select_action(self, state, epsilon=0.2):
        """选择动作（添加探索机制）"""
        if random.random() < epsilon:
            return self._random_action(state)
        else:
            return self._model_based_action(state)

    def _model_based_action(self, state):
        """模型预测选择"""
        with torch.no_grad():
            # 1. 基础特征提取
            features = self.forward(state['player_features'])
            
            # 2. 如果启用协同机制,使用GAT更新特征
            if self.use_synergy:
                # 获取协同矩阵
                synergy_matrix = state['raw_features'].get('synergy_matrix', None)
                if synergy_matrix is not None:
                    # 准备输入
                    features_for_gat = features.squeeze(0).unsqueeze(0)
                    synergy_tensor = torch.from_numpy(synergy_matrix).float().to(self.device)
                    adj_matrix = (synergy_tensor > 0).float()
                    
                    # 获取位置信息用于位置编码
                    position_ids = state['position_onehot'].argmax(dim=1).unsqueeze(0)
                    
                    # GAT处理
                    features = self.gat_fusion(
                        features_for_gat, 
                        adj_matrix.unsqueeze(0),
                        position_ids
                    )
            
            # 3. 计算选择概率
            action_probs = torch.sigmoid(self.action_head(features)).squeeze(0)
            
            # 4. 根据设置选择决策方法
            if self.use_ilp:
                selected = self._select_by_ilp(action_probs, state)
            else:
                selected = self._select_by_position_and_budget(action_probs, state)
            
            return selected.cpu().numpy()

    def _select_by_position_and_budget(self, action_probs, state):
        """
        改为通过ILP解算器进行组合选择
        """
        values = state['raw_features']['values']  # 每名球员的价格
        teams = state['raw_features'].get('names', np.array([]))  # 这里可以放球队信息
        # 如果存储了球队名称，可以记录 team 列: teams[i] = "Chelsea"
        # 注意: 若要区分同名球队，需要先做预处理，让 teams 存真正的队伍ID或简写

        # 调用解算器
        selected_array = self.ilp_solver.solve(action_probs, values, teams)

        return selected_array

    def _select_by_ilp(self, action_probs, state):
        values = state['raw_features']['values']
        teams = state['raw_features'].get('names', np.array([]))
        return self.ilp_solver.solve(action_probs, values, teams)

    def _random_action(self, state):
        """随机选择符合预算的球员"""
        values = state['raw_features']['values']
        positions = state['position_onehot'].argmax(dim=1)
        
        # 按位置分组
        position_groups = {
            0: [],  # GK
            1: [],  # DEF
            2: [],  # MID
            3: []   # FWD
        }
        
        for i in range(len(positions)):
            position_groups[positions[i].item()].append(i)
        
        # 随机选择符合位置要求的球员
        selected = np.zeros(len(values))
        for pos, indices in position_groups.items():
            k = 2 if pos == 0 else 5 if pos in [1,2] else 3
            if len(indices) >= k:
                chosen = np.random.choice(indices, k, replace=False)
                selected[chosen] = 1
        
        # 确保总价值不超过预算
        total_cost = (selected * values.cpu().numpy()).sum()
        while total_cost > self.budget and np.sum(selected) > 0:
            # 移除最贵的球员
            selected[np.argmax(selected * values.cpu().numpy())] = 0
            total_cost = (selected * values.cpu().numpy()).sum()
        
        return selected

    def _count_positions(self, action, state):
        """统计每个位置选择的球员数量"""
        positions = ['GK', 'DEF', 'MID', 'FWD']
        counts = {}
        for i, pos in enumerate(positions):
            pos_mask = (state['position_onehot'][:, i] == 1)
            counts[pos] = (action * pos_mask).sum().item()
        return counts
    
    def get_representation(self, state):
        """获取状态表示"""
        if isinstance(state, dict):
            state = state['player_features']
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
            
        return self.forward(state)
    
    def compute_loss(self, batch, current_season):
        """计算强化学习损失"""
        try:
            states = torch.stack([s['player_features'] for s in batch['states']]).to(self.device)
            actions = torch.FloatTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).to(self.device)
            
            # 1. 策略网络损失
            action_probs = torch.sigmoid(self.action_head(self.forward(states)))
            policy_loss = F.binary_cross_entropy_with_logits(
                action_probs, 
                actions,
                reduction='none'
            )
            policy_loss = (policy_loss * rewards.unsqueeze(-1)).mean()
            
            # 2. 价值网络损失
            values = self.value_head(self.forward(states)).squeeze()
            value_loss = F.mse_loss(values, rewards)
            
            # 3. 正则化
            l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.parameters())
            
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss + l2_reg
            
            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total_loss': total_loss,
                'l2_reg': l2_reg.item()
            }
            
        except Exception as e:
            print(f"Error in compute_loss: {str(e)}")
            return None

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