import torch
import torch.nn as nn

class STConv(nn.Module):
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
        
        # ... [中间代码保持不变] ...
        
        # 验证输出维度
        output = self.fusion(combined_features)
        assert output.size(-1) == self.hidden_dim, f"STConv 输出维度错误：{output.shape}"
        assert output.size(1) == num_players, f"球员数量不匹配：{output.shape}"
        
        return output