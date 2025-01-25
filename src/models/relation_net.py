import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationNet(nn.Module):
    """动态异构图构建的关系学习网络"""
    def __init__(self, input_dim=12):
        super().__init__()
        self.input_dim = input_dim
        
        # 位置依赖关系学习
        self.position_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 球队协作关系学习
        self.team_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 价值相似关系学习
        self.value_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 关系融合层
        self.fusion = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def compute_position_relation(self, x_i, x_j):
        """计算位置依赖关系"""
        combined = torch.cat([x_i, x_j], dim=-1)
        return self.position_net(combined)
    
    def compute_team_relation(self, x_i, x_j):
        """计算球队协作关系"""
        combined = torch.cat([x_i, x_j], dim=-1)
        return self.team_net(combined)
    
    def compute_value_relation(self, x_i, x_j):
        """计算价值相似关系"""
        combined = torch.cat([x_i, x_j], dim=-1)
        return self.value_net(combined)
    
    def forward(self, x):
        """
        输入: x [batch_size, num_players, input_dim]
        输出: adj_matrices [batch_size, 3, num_players, num_players]
        """
        batch_size, num_players, _ = x.shape
        device = x.device
        
        # 初始化三种关系矩阵
        position_adj = torch.zeros(batch_size, num_players, num_players, device=device)
        team_adj = torch.zeros_like(position_adj)
        value_adj = torch.zeros_like(position_adj)
        
        # 计算所有球员对之间的关系
        for i in range(num_players):
            for j in range(i+1, num_players):
                x_i = x[:, i, :]
                x_j = x[:, j, :]
                
                # 计算三种关系
                pos_rel = self.compute_position_relation(x_i, x_j)
                team_rel = self.compute_team_relation(x_i, x_j)
                val_rel = self.compute_value_relation(x_i, x_j)
                
                # 融合关系
                relations = torch.cat([pos_rel, team_rel, val_rel], dim=-1)
                fused_rel = self.fusion(relations)
                
                # 填充邻接矩阵(对称)
                position_adj[:, i, j] = position_adj[:, j, i] = pos_rel.squeeze(-1)
                team_adj[:, i, j] = team_adj[:, j, i] = team_rel.squeeze(-1)
                value_adj[:, i, j] = value_adj[:, j, i] = val_rel.squeeze(-1)
        
        # 归一化邻接矩阵
        position_adj = F.softmax(position_adj, dim=-1)
        team_adj = F.softmax(team_adj, dim=-1)
        value_adj = F.softmax(value_adj, dim=-1)
        
        return torch.stack([position_adj, team_adj, value_adj], dim=1) 