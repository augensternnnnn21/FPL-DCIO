import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class StrategyGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 历史策略编码器
        self.history_encoder = nn.ModuleDict({
            'performance': self._build_performance_encoder(),
            'advanced': self._build_advanced_encoder(),
            'transfers': self._build_transfer_encoder()
        })
        
        # 2. 当前环境编码器
        self.context_encoder = nn.ModuleDict({
            'match': self._build_match_encoder(),
            'player': self._build_player_encoder(),
            'team': self._build_team_encoder()
        })
        
        # 3. 时序注意力层
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.model.strategy_hidden_dim,
            num_heads=8,
            dropout=config.model.strategy_dropout
        )
        
        # 4. 策略生成器
        self.strategy_generator = nn.Sequential(
            nn.Linear(config.model.strategy_hidden_dim * 2, config.model.strategy_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.strategy_dropout),
            nn.Linear(config.model.strategy_hidden_dim, config.num_strategies)
        )
        
    def _build_performance_encoder(self):
        """构建球员表现编码器"""
        return nn.Sequential(
            nn.Linear(4, self.config.model.strategy_hidden_dim // 2),  # form, minutes_last5, avg_points, trend
            nn.ReLU(),
            nn.Dropout(self.config.model.strategy_dropout),
            nn.Linear(self.config.model.strategy_hidden_dim // 2, self.config.model.strategy_hidden_dim)
        )
    
    def _build_advanced_encoder(self):
        """构建高级统计编码器"""
        return nn.Sequential(
            nn.Linear(14, self.config.model.strategy_hidden_dim // 2),  # 基础表现特征
            nn.ReLU(),
            nn.Dropout(self.config.model.strategy_dropout),
            nn.Linear(self.config.model.strategy_hidden_dim // 2, self.config.model.strategy_hidden_dim)
        )
    
    def _build_transfer_encoder(self):
        """构建转会数据编码器"""
        return nn.Sequential(
            nn.Linear(5, self.config.model.strategy_hidden_dim // 2),  # 转会相关特征
            nn.ReLU(),
            nn.Dropout(self.config.model.strategy_dropout),
            nn.Linear(self.config.model.strategy_hidden_dim // 2, self.config.model.strategy_hidden_dim)
        )
    
    def _build_match_encoder(self):
        """构建比赛信息编码器"""
        return nn.Sequential(
            nn.Linear(2, self.config.model.strategy_hidden_dim // 2),  # opponent_team, was_home
            nn.ReLU(),
            nn.Dropout(self.config.model.strategy_dropout),
            nn.Linear(self.config.model.strategy_hidden_dim // 2, self.config.model.strategy_hidden_dim)
        )
    
    def _build_player_encoder(self):
        """构建球员状态编码器"""
        return nn.Sequential(
            nn.Linear(3, self.config.model.strategy_hidden_dim // 2),  # minutes_ma5, form, total_points_ma5
            nn.ReLU(),
            nn.Dropout(self.config.model.strategy_dropout),
            nn.Linear(self.config.model.strategy_hidden_dim // 2, self.config.model.strategy_hidden_dim)
        )
    
    def _build_team_encoder(self):
        """构建球队状态编码器"""
        return nn.Sequential(
            nn.Linear(2, self.config.model.strategy_hidden_dim // 2),  # team_strength, opponent_strength
            nn.ReLU(),
            nn.Dropout(self.config.model.strategy_dropout),
            nn.Linear(self.config.model.strategy_hidden_dim // 2, self.config.model.strategy_hidden_dim)
        )
        
    def forward(self, features):
        # 1. 编码历史策略
        history_encodings = []
        
        # 编码性能数据
        performance_features = features['historical']['performance']
        performance_encoding = self.history_encoder['performance'](performance_features)
        history_encodings.append(performance_encoding)
        
        # 编码球员统计
        player_stats = features['historical']['player_stats']
        advanced_encoding = self.history_encoder['advanced'](player_stats)
        history_encodings.append(advanced_encoding)
        
        # 编码转会数据
        transfer_features = features['historical']['transfers']
        transfer_encoding = self.history_encoder['transfers'](transfer_features)
        history_encodings.append(transfer_encoding)
        
        history_tensor = torch.stack(history_encodings, dim=1)
        
        # 2. 编码当前环境
        context_encodings = []
        
        # 编码比赛信息
        match_features = features['context']['fixture']
        match_encoding = self.context_encoder['match'](match_features)
        context_encodings.append(match_encoding)
        
        # 编码球员状态
        player_features = features['context']['player_state']
        player_encoding = self.context_encoder['player'](player_features)
        context_encodings.append(player_encoding)
        
        # 编码球队状态
        team_features = features['context']['team_state']
        team_encoding = self.context_encoder['team'](team_features)
        context_encodings.append(team_encoding)
        
        context_tensor = torch.stack(context_encodings, dim=1)
        
        # 3. 应用时序注意力
        attended_history, attention_weights = self.temporal_attention(
            history_tensor, history_tensor, history_tensor
        )
        
        # 4. 融合历史和当前信息
        history_summary = attended_history.mean(dim=1)  # 历史信息汇总
        context_summary = context_tensor.mean(dim=1)  # 当前环境信息汇总
        fused_features = torch.cat([history_summary, context_summary], dim=1)
        
        # 5. 生成策略
        strategy_logits = self.strategy_generator(fused_features)
        
        return {
            'strategy_logits': strategy_logits,
            'attention_weights': attention_weights,
            'history_encoding': history_summary,
            'context_encoding': context_summary
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x) 