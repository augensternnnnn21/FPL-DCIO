import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GATFusion(nn.Module):
    """图注意力融合模块,用于建模球员间的协同关系"""
    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多头图注意力层
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim//num_heads, dropout)
            for _ in range(num_heads)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
    def forward(self, x, adj_matrix, position_ids=None):
        """
        Args:
            x: [batch_size, num_players, input_dim] 球员特征
            adj_matrix: [batch_size, num_players, num_players] 邻接矩阵
            position_ids: [batch_size, num_players] 位置编码(可选)
        """
        batch_size, num_players = x.size(0), x.size(1)
        
        # 1. 特征转换
        h = self.feature_transform(x)
        
        # 2. 添加位置编码
        if position_ids is not None:
            h = h + self.pos_encoder(position_ids)
            
        # 3. 多头注意力处理
        gat_outputs = torch.stack([
            gat(h, adj_matrix) for gat in self.gat_layers
        ], dim=-1)  # [batch_size, num_players, hidden_dim//num_heads, num_heads]
        
        # 重塑并拼接
        h = gat_outputs.reshape(batch_size, num_players, -1)  # [batch_size, num_players, hidden_dim]
        
        # 5. 输出投影
        out = self.output_proj(h)
        
        # 6. 残差连接
        out = out + x
        
        return out

class GraphAttentionLayer(nn.Module):
    """单个图注意力层"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        # 注意力参数
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        
        # LeakyReLU激活
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, h, adj):
        # 1. 线性变换
        Wh = self.W(h)  # [batch_size, num_players, out_dim]
        
        # 2. 计算注意力分数
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(self.a(a_input))
        
        # 3. Mask注意力分数
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # 4. Softmax归一化
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout)
        
        # 5. 聚合邻居信息
        h_prime = torch.bmm(attention, Wh)
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        B, N, E = Wh.shape  # batch_size, num_players, out_dim
        
        # 复制特征用于计算注意力
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # [B, N*N, E]
        Wh_repeated_alternating = Wh.repeat(1, N, 1)  # [B, N*N, E]
        
        # 拼接特征
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(B, N, N, 2*E)

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 