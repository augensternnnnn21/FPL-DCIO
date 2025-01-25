import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from .dst_meta_grl import DynamicSpatioTemporalMetaLearner

class NoPositionModel(nn.Module):
    """无位置感知的模型变体"""
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_PLAYERS)
        )
    
    def forward(self, x):
        return self.network(x)

class NoHierarchyModel(nn.Module):
    """无分层决策的模型变体"""
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU()
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=HIDDEN_DIM,
            num_heads=NUM_ATTENTION_HEADS,
            dropout=DROPOUT_RATE
        )
        
        self.output = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_PLAYERS)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(0, 1)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = x.transpose(0, 1)
        return self.output(x.mean(dim=1))

class NoMetaModel(nn.Module):
    """无元学习的模型变体"""
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU()
        )
        
        self.position_embedding = nn.Embedding(4, HIDDEN_DIM)
        
        self.policy = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_PLAYERS)
        )
    
    def forward(self, x, positions):
        x = self.encoder(x)
        pos_embed = self.position_embedding(positions)
        x = x + pos_embed
        return self.policy(x)

class AblationModel(DynamicSpatioTemporalMetaLearner):
    """用于消融实验的模型变体"""
    def __init__(self, input_dim=12, hidden_dim=64, ablation_config=None):
        super().__init__(input_dim, hidden_dim)
        self.ablation_config = ablation_config or {}
        
        # 根据消融配置修改模型组件
        if self.ablation_config.get('disable_dynamic_graph', False):
            # 使用固定图结构
            self.relation_learner = self._create_fixed_relation_net()
        
        if self.ablation_config.get('disable_meta_learning', False):
            # 禁用元学习组件
            self.meta_distiller = None
            self.curriculum = None
            self.grad_aggregator = None
        
        if self.ablation_config.get('disable_curriculum', False):
            # 禁用课程学习
            self.curriculum = None
        
        if self.ablation_config.get('hard_constraints', False):
            # 使用硬约束而不是可微分约束
            self.constraint_checker = self._create_hard_constraint_checker()
    
    def _create_fixed_relation_net(self):
        """创建使用固定图结构的关系网络"""
        class FixedRelationNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fixed_adj = None
            
            def forward(self, x):
                batch_size = x.shape[0]
                if self.fixed_adj is None:
                    # 创建固定的邻接矩阵
                    num_players = x.shape[1]
                    self.fixed_adj = torch.eye(num_players).unsqueeze(0).repeat(3, 1, 1)
                    self.fixed_adj = self.fixed_adj.to(x.device)
                return self.fixed_adj.expand(batch_size, -1, -1, -1)
        
        return FixedRelationNet()
    
    def _create_hard_constraint_checker(self):
        """创建使用硬约束的检查器"""
        class HardConstraintChecker(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, selections, player_data):
                violations = {}
                
                # 预算约束
                total_cost = (selections * player_data['value']).sum(dim=1)
                violations['budget'] = (total_cost > 100.0).float()
                
                # 位置约束
                position_counts = (selections * player_data['position_onehot']).sum(dim=1)
                required_counts = torch.tensor([2, 5, 5, 3], device=selections.device)
                violations['position'] = (position_counts != required_counts).any(dim=1).float()
                
                # 球队约束
                team_counts = (selections * player_data['team_onehot']).sum(dim=1)
                violations['team'] = (team_counts > 3).any(dim=1).float()
                
                return {
                    'total': sum(violations.values()),
                    'details': violations
                }
        
        return HardConstraintChecker()
    
    def forward(self, state, meta_knowledge=None, constraints=None):
        """根据消融配置修改前向传播"""
        if self.ablation_config.get('disable_meta_learning', False):
            meta_knowledge = None
        
        if self.ablation_config.get('disable_curriculum', False):
            constraints = None
        
        return super().forward(state, meta_knowledge, constraints)

def create_ablation_variants():
    """创建不同的消融实验变体"""
    variants = {
        'full_model': AblationModel(),
        'no_dynamic_graph': AblationModel(
            ablation_config={'disable_dynamic_graph': True}
        ),
        'no_meta_learning': AblationModel(
            ablation_config={'disable_meta_learning': True}
        ),
        'no_curriculum': AblationModel(
            ablation_config={'disable_curriculum': True}
        ),
        'hard_constraints': AblationModel(
            ablation_config={'hard_constraints': True}
        ),
        'minimal': AblationModel(
            ablation_config={
                'disable_dynamic_graph': True,
                'disable_meta_learning': True,
                'disable_curriculum': True,
                'hard_constraints': True
            }
        )
    }
    return variants