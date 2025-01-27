import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_PLAYERS

HIDDEN_DIM = 64  # 设置隐藏层维度

class GNNLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 使用正交初始化
        self.linear = nn.Linear(input_dim, input_dim)
        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.1)
        
        self.agg = nn.Linear(input_dim * 2, input_dim)
        nn.init.orthogonal_(self.agg.weight)
        nn.init.constant_(self.agg.bias, 0.1)

        self.attention = nn.Linear(input_dim * 2, 1)  # 添加注意力机制
        nn.init.orthogonal_(self.attention.weight)
        nn.init.constant_(self.attention.bias, 0.1)

    def forward(self, x, adj):
        """
        x: [batch_size, num_nodes, features]
        adj: [batch_size, num_nodes, num_nodes] or [num_nodes, num_nodes]
        """
        if len(adj.shape) == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        
        h = self.linear(x)
        neighbor_h = torch.bmm(adj.float(), h)
         # 计算注意力权重
        combined = torch.cat([h, neighbor_h], dim=-1)
        attention_scores = F.softmax(self.attention(combined), dim=1)
        weighted_neighbor = attention_scores * neighbor_h
        # 添加位置注意力权重
        pos_attention = torch.matmul(x, x.transpose(1,2)) 
        adj = adj * pos_attention  # 通过位置相似性加权邻接矩阵
        
        return F.relu(h + weighted_neighbor)

class ConstraintAwareGNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.pos_encoder = nn.Embedding(4, input_dim)  # 位置编码
        self.attention = nn.MultiheadAttention(input_dim, num_heads=4)
        
    def forward(self, x, adj, pos_masks):
        # 位置增强的特征
        pos_features = self.pos_encoder(pos_masks.argmax(dim=1))
        x = x + pos_features
        
        # 约束感知的注意力
        attn_output, _ = self.attention(
            x, x, x, 
            key_padding_mask=~pos_masks.bool()  # 屏蔽非法位置
        )
        return F.relu(attn_output)

class DCIO(nn.Module):
    def __init__(self, adj_matrix, player_data):  # 新增 player_data 参数
        super().__init__()
        if len(adj_matrix.shape) == 3:
            adj_matrix = adj_matrix.squeeze(0)
        self.adj = adj_matrix
        self.data = player_data  # 存储球员数据
        self.num_players = len(player_data['name'].unique())  # 从数据中获取实际球员数量
        self.input_dim = 5
        self.hidden_dim = HIDDEN_DIM
        
        self.position_embedding = nn.Embedding(4, self.hidden_dim)
        
        # 主网络
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)
        nn.init.orthogonal_(self.embedding.weight)  # 参数初始化
        nn.init.constant_(self.embedding.bias, 0.1)
        self.gnn1 = GNNLayer(self.hidden_dim)
        self.gnn2 = GNNLayer(self.hidden_dim)
        self.q_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_PLAYERS)
        )
        
        # 目标网络
        self.target_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.target_gnn1 = GNNLayer(self.hidden_dim)
        self.target_gnn2 = GNNLayer(self.hidden_dim)
        self.target_q_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_PLAYERS)
        )
        
        self.update_target_network()
    
    def _get_position_index(self, player_idx):
        """通过存储的球员数据获取位置索引"""
        unique_players = self.data['name'].unique()
        player_name = unique_players[player_idx]
        pos = self.data[self.data['name'] == player_name]['position'].iloc[0]
        return {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}[pos]

    def update_target_network(self, tau=0.005):
        """软更新目标网络"""
        for target_param, param in zip(self.target_embedding.parameters(), self.embedding.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_gnn1.parameters(), self.gnn1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_gnn2.parameters(), self.gnn2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def forward(self, state, use_target=False):
        """使用主网络或目标网络进行前向传播"""
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        
        # 使用主网络或目标网络
        if not use_target:
            x = self.embedding(state)
            x = self.gnn1(x, self.adj)
            x = self.gnn2(x, self.adj)
            x = x.mean(dim=1)
            q_values = self.q_net(x)
        else:
            with torch.no_grad():
                x = self.target_embedding(state)
                x = self.target_gnn1(x, self.adj)
                x = self.target_gnn2(x, self.adj)
                x = x.mean(dim=1)
                q_values = self.target_q_net(x)
        
        return q_values  # 直接返回 [batch, num_players] 的 Q 值
        