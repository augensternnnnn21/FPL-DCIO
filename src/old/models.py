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
    def __init__(self, adj, data, hidden_dim=128):
        super().__init__()
        # 保持原有的模型结构
        self.adj = adj
        self.data = data
        self.hidden_dim = hidden_dim
        
        # 原有的层
        self.gnn1 = GNNLayer(data.shape[1])
        self.gnn2 = GNNLayer(hidden_dim)
        self.gat = GNNLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        
        # 添加MAML相关的组件
        self.inner_lr = 0.01  # 内循环学习率
        self.adaptation_steps = 3  # 快速适应步数
        
    def forward(self, x, adapt_params=None):
        """
        前向传播，支持MAML的快速适应
        Args:
            x: 输入特征
            adapt_params: 适应后的参数(用于MAML)
        """
        if adapt_params is None:
            # 使用当前模型参数
            h = F.relu(self.gnn1(x, self.adj))
            h = F.relu(self.gnn2(h, self.adj))
            h = F.relu(self.gat(h, self.adj))
            out = self.fc(h)
        else:
            # 使用适应后的参数
            h = F.relu(F.linear(x, adapt_params['gnn1.linear.weight'], adapt_params['gnn1.linear.bias']))
            h = F.relu(F.linear(h, adapt_params['gnn2.linear.weight'], adapt_params['gnn2.linear.bias']))
            h = F.relu(F.linear(h, adapt_params['gat.linear.weight'], adapt_params['gat.linear.bias']))
            out = F.linear(h, adapt_params['fc.weight'], adapt_params['fc.bias'])
        
        return out
    
    def adapt(self, support_x, support_y, first_order=True):
        """
        在support set上快速适应
        Args:
            support_x: 支持集特征
            support_y: 支持集标签
            first_order: 是否使用一阶近似
        """
        params = {name: param.clone() for name, param in self.named_parameters()}
        
        for _ in range(self.adaptation_steps):
            pred = self.forward(support_x, params)
            loss = F.mse_loss(pred, support_y)
            
            grads = torch.autograd.grad(loss, params.values(),
                                      create_graph=not first_order)
            
            # 更新参数
            for (name, param), grad in zip(params.items(), grads):
                params[name] = param - self.inner_lr * grad
        
        return params
        