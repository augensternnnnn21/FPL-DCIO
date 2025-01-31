import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveInfoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征转换层
        self.feature_transforms = nn.ModuleDict({
            'historical_strategy': nn.Linear(23, config.model.info_hidden_dim),  # 14 + 4 + 5
            'current_context': nn.Linear(7, config.model.info_hidden_dim)  # 2 + 3 + 2
        })
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=config.model.info_hidden_dim,
            num_heads=config.model.info_num_heads,
            dropout=config.model.info_dropout
        )
        
        # 特征权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(config.model.info_hidden_dim, config.model.info_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.model.info_dropout),
            nn.Linear(config.model.info_hidden_dim // 2, 1)
        )
        
    def forward(self, features_dict):
        # 1. 转换特征
        transformed_features = {}
        for name, feature in features_dict.items():
            transformed = self.feature_transforms[name](feature)
            transformed_features[name] = transformed
        
        # 2. 堆叠特征
        feature_stack = torch.stack(list(transformed_features.values()), dim=0)
        
        # 3. 应用注意力机制
        attended_features, attention_weights = self.attention(
            feature_stack, feature_stack, feature_stack
        )
        
        # 4. 生成特征权重
        feature_weights = torch.softmax(
            self.weight_generator(attended_features).squeeze(-1),
            dim=0
        )
        
        # 5. 加权组合特征
        weighted_features = (attended_features * feature_weights.unsqueeze(-1)).sum(dim=0)
        
        return weighted_features, feature_weights, attention_weights
    
    def get_importance_scores(self, features_dict):
        """获取特征重要性分数"""
        with torch.no_grad():
            _, feature_weights, attention_weights = self.forward(features_dict)
            return {
                'feature_weights': feature_weights,
                'attention_weights': attention_weights
            } 