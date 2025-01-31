import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaStrategyNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 计算输入维度
        self.input_dim = sum(config.feature_dims.values())
        
        # 构建MLP层
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in config.model.meta_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.model.meta_dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 策略输出层
        self.mlp = nn.Sequential(*layers)
        self.strategy_head = nn.Linear(prev_dim, config.num_strategies)
        
        # 策略嵌入层
        self.strategy_embedding = nn.Parameter(
            torch.randn(config.num_strategies, prev_dim)
        )
        
    def forward(self, features_dict):
        # 特征拼接
        feature_list = []
        for name in sorted(features_dict.keys()):  # 确保特征顺序一致
            feature_list.append(features_dict[name])
        x = torch.cat(feature_list, dim=-1)
        
        # MLP特征提取
        hidden_features = self.mlp(x)
        
        # 策略logits计算
        strategy_logits = self.strategy_head(hidden_features)
        
        # 计算与策略原型的相似度
        strategy_similarities = F.cosine_similarity(
            hidden_features.unsqueeze(1),  # [B, 1, H]
            self.strategy_embedding.unsqueeze(0),  # [1, N, H]
            dim=2
        )
        
        return {
            'strategy_logits': strategy_logits,
            'strategy_similarities': strategy_similarities,
            'hidden_features': hidden_features
        }
    
    def get_strategy_embedding(self, strategy_idx):
        """获取特定策略的嵌入表示"""
        return self.strategy_embedding[strategy_idx]
    
    def compute_strategy_distances(self):
        """计算策略之间的相似度矩阵"""
        return F.cosine_similarity(
            self.strategy_embedding.unsqueeze(1),
            self.strategy_embedding.unsqueeze(0),
            dim=2
        ) 