import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedTCN(nn.Module):
    """带有扩张的时间卷积网络"""
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3, dilation=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # 添加padding以保持输入输出维度一致
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=input_dim,
            kernel_size=1,
            padding=0
        )
        
        # 修改 LayerNorm 的维度设置
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch_size, num_players, input_dim]
        """
        # 调整维度顺序
        x = x.transpose(1, 2)  # [batch_size, input_dim, num_players]
        
        # 保存输入用于残差连接
        identity = x
        
        # 第一次卷积
        out = self.conv1(x)
        
        # 转换维度进行归一化
        out = out.transpose(1, 2)  # [batch_size, num_players, hidden_dim]
        out = self.norm(out)
        out = out.transpose(1, 2)  # [batch_size, hidden_dim, num_players]
        
        # 激活函数
        out = self.activation(out)
        
        # 第二次卷积
        out = self.conv2(out)
        
        # 确保维度匹配
        if out.size() != identity.size():
            identity = F.pad(
                identity,
                (0, out.size(-1) - identity.size(-1)),
                mode='replicate'
            )
        
        # 残差连接
        out = out + identity
        
        # 恢复维度顺序
        out = out.transpose(1, 2)  # [batch_size, num_players, input_dim]
        
        return out

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

class STConv(nn.Module):
    """时空卷积模块"""
    def __init__(self, input_dim, hidden_dim, gcn_types, temporal_win=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temporal_win = temporal_win
        
        # 时间卷积网络
        self.tcn = DilatedTCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=3,
            dilation=2
        )
        
        # 图注意力层
        self.gat_layers = nn.ModuleDict({
            gtype: GATFusion(input_dim=input_dim, hidden_dim=hidden_dim)
            for gtype in gcn_types
        })
        
        # 时间特征投影
        self.temporal_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * (len(gcn_types) + 1), hidden_dim),  # +1 是为了时间特征
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x, adj_matrices):
        """
        前向传播
        Args:
            x: [batch_size, temporal_win, num_players, input_dim]
            adj_matrices: Dict[str, Tensor] 不同类型的邻接矩阵
        """
        batch_size = x.size(0)
        num_players = x.size(2)

        # 验证输入维度
        assert num_players > 1, f"球员数量不足：{num_players}"
        
        # 1. 时间卷积
        temporal_features = []
        for t in range(self.temporal_win):
            temp_feat = self.tcn(x[:, t])  # [batch_size, num_players, input_dim]
            temporal_features.append(temp_feat)
        
        # 融合时间特征
        temporal_features = torch.stack(temporal_features, dim=1)  # [batch_size, temporal_win, num_players, input_dim]
        temporal_features = temporal_features.mean(dim=1)  # [batch_size, num_players, input_dim]
        
        # 投影时间特征到隐藏维度
        temporal_features = self.temporal_proj(temporal_features)  # [batch_size, num_players, hidden_dim]
        
        # 2. 图注意力
        graph_features = []
        for gtype, adj_matrix in adj_matrices.items():
            if gtype in self.gat_layers:
                gat_out = self.gat_layers[gtype](temporal_features, adj_matrix)  # [batch_size, num_players, hidden_dim]
                graph_features.append(gat_out)
        
        # 3. 特征融合
        graph_features.append(temporal_features)  # 添加时间特征
        combined_features = torch.cat(graph_features, dim=-1)  # [batch_size, num_players, hidden_dim * (num_types + 1)]
        output = self.fusion(combined_features)  # [batch_size, num_players, hidden_dim]

        # 验证输出维度
        output = self.fusion(combined_features)
        assert output.size(-1) == self.hidden_dim, f"STConv 输出维度错误：{output.shape}"
        assert output.size(1) == num_players, f"球员数量不匹配：{output.shape}"
        
        return output 