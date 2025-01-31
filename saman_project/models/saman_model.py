import torch
import torch.nn as nn
import torch.nn.functional as F

from .strategy_generator import StrategyGenerator
from .adaptive_info_model import AdaptiveInfoModel

class SAMANModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 策略生成器
        self.strategy_generator = StrategyGenerator(config)
        
        # 自适应信息模型
        self.info_model = AdaptiveInfoModel(config)
        
        # 元策略网络
        meta_layers = []
        input_dim = config.model.info_hidden_dim
        for hidden_dim in config.model.meta_hidden_dims:
            meta_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.model.meta_dropout)
            ])
            input_dim = hidden_dim
        meta_layers.append(nn.Linear(input_dim, 1))  # 输出单个预测值
        self.meta_network = nn.Sequential(*meta_layers)
        
    def forward(self, features):
        # 1. 从策略生成器获取策略
        strategy_output = self.strategy_generator(features)
        
        # 2. 从自适应信息模型获取加权特征
        weighted_features, feature_weights, attention_weights = self.info_model({
            'historical_strategy': torch.cat([
                features['historical']['performance'],
                features['historical']['player_stats'],
                features['historical']['transfers']
            ], dim=-1),
            'current_context': torch.cat([
                features['context']['fixture'],
                features['context']['player_state'],
                features['context']['team_state']
            ], dim=-1)
        })
        
        # 3. 使用元策略网络生成最终预测
        prediction = self.meta_network(weighted_features)
        
        # 确保输出维度为 [batch_size, 1]
        return prediction.view(-1, 1)
    
    def compute_loss(self, outputs, targets, return_details=False):
        """计算模型损失"""
        if not isinstance(outputs, dict):
            strategy_loss = F.cross_entropy(outputs, targets)
            return strategy_loss
            
        # 策略预测损失
        strategy_loss = F.cross_entropy(
            outputs['final_strategy_logits'], 
            targets
        )
        
        # 信息正则化损失
        info_loss = self._compute_info_regularization(
            outputs['feature_weights'],
            outputs['attention_weights']
        )
        
        # 策略一致性损失
        consistency_loss = self._compute_consistency_loss(
            outputs['strategy_logits'],
            outputs['meta_strategy_logits'],
            outputs['strategy_similarities']
        )
        
        # 总损失
        total_loss = (
            strategy_loss + 
            self.config.training.info_reg_weight * info_loss +
            0.1 * consistency_loss  # 可配置权重
        )
        
        if not return_details:
            return total_loss
            
        return {
            'total_loss': total_loss,
            'strategy_loss': strategy_loss,
            'info_loss': info_loss,
            'consistency_loss': consistency_loss
        }
    
    def _compute_info_regularization(self, feature_weights, attention_weights):
        """计算信息正则化损失"""
        # 特征权重的熵
        weight_entropy = -(feature_weights * torch.log(feature_weights + 1e-10)).sum(1).mean()
        
        # 注意力权重的熵
        attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(1).mean()
        
        return weight_entropy + attention_entropy
    
    def _compute_consistency_loss(self, strategy_logits, meta_logits, similarities):
        """计算策略一致性损失"""
        # KL散度
        kl_loss = F.kl_div(
            F.log_softmax(strategy_logits, dim=1),
            F.softmax(meta_logits, dim=1),
            reduction='batchmean'
        )
        
        # 相似度一致性
        sim_loss = F.mse_loss(
            F.softmax(strategy_logits, dim=1),
            F.softmax(similarities, dim=1)
        )
        
        return kl_loss + sim_loss
        
    def get_interpretable_weights(self, features_dict):
        """获取可解释的特征权重"""
        with torch.no_grad():
            outputs = self.forward(features_dict, return_details=True)
            return {
                'feature_importance': outputs['feature_weights'],
                'attention_patterns': outputs['attention_weights'],
                'strategy_similarity': outputs['strategy_similarities']
            } 