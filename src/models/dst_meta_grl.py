import torch
import torch.nn as nn
import torch.nn.functional as F
from .relation_net import RelationNet
from .st_conv import STConv
from .meta_learning import MetaDistiller, CurriculumScheduler, GradientAggregator
from .constraints import DifferentiableChecker

class DynamicSpatioTemporalMetaLearner(nn.Module):
    """动态时空元图强化学习模型"""
    def __init__(self, input_dim=12, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1. 动态异构图构建层
        self.relation_learner = RelationNet(input_dim=input_dim)
        
        # 2. 时空融合模块
        self.st_conv = STConv(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gcn_types=['position', 'team', 'value'],
            temporal_win=5
        )
        
        # 3. 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4. 元学习组件
        self.curriculum = CurriculumScheduler()
        self.grad_aggregator = GradientAggregator()
        self.meta_distiller = MetaDistiller(
            inner_loop=self.curriculum,
            outer_loop=self.grad_aggregator
        )
        
        # 5. 约束处理器
        self.constraint_checker = DifferentiableChecker()
        
        # 6. 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # 7. 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        
        # 8. 动作生成器
        self.action_generator = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 600)  # 假设最大球员数为600
        )
        
        # 修改这部分，延迟导入泛化组件
        from .generalization import DomainAdversarialTraining, SeasonalAdaptationNetwork
        self.domain_adversarial = DomainAdversarialTraining(feature_dim=hidden_dim)
        self.seasonal_adaptation = SeasonalAdaptationNetwork(hidden_dim)
        
        # 添加赛季编码器
        self.season_encoder = nn.Sequential(
            nn.Linear(4, 16),  # 4个季节特征：年份、是否COVID赛季等
            nn.ReLU(),
            nn.LayerNorm(16)
        )
    
    def get_representation(self, state):
        """获取状态表示，用于泛化性评估"""
        with torch.no_grad():
            output = self.forward(state)
            return output['domain_features']
    
    def forward(self, state, meta_knowledge=None, constraints=None, season_info=None):
        """
        前向传播
        Args:
            state: 字典，包含以下键：
                - 'features': [batch_size, num_players, feature_dim]
                - 'position_onehot': [batch_size, num_players, 4]
                - 'team_onehot': [batch_size, num_players, 20]
                - 'values': [batch_size, num_players]
                - 'history': [batch_size, 5, num_players, feature_dim]
            meta_knowledge: 元知识
            constraints: 约束列表
            season_info: 赛季信息
        """
        # 确保输入是字典
        if not isinstance(state, dict):
            raise TypeError(f"Expected state to be dict, got {type(state)}")
        
        # 获取特征
        features = state['features']
        position_onehot = state['position_onehot']
        team_onehot = state['team_onehot']
        values = state['values']
        history = state['history']
        
        # 获取批次大小
        if len(features.shape) == 2:
            # 添加批次维度
            features = features.unsqueeze(0)
            position_onehot = position_onehot.unsqueeze(0)
            team_onehot = team_onehot.unsqueeze(0)
            values = values.unsqueeze(0)
            history = history.unsqueeze(0)
        
        batch_size = features.shape[0]
        device = features.device
        
        # 1. 构建动态关系图
        adj_matrices = self.relation_learner(features)
        
        # 构建不同类型的邻接矩阵字典
        adj_dict = {
            'position': self._build_position_adj_matrix(position_onehot),
            'team': self._build_team_adj_matrix(team_onehot),
            'value': self._build_value_adj_matrix(values)
        }
        
        # 2. 时空特征融合
        spatio_temporal_features = self.st_conv(
            history,
            adj_dict
        )  # [batch_size, num_players, hidden_dim]
        assert spatio_temporal_features.size(-1) == self.hidden_dim, \
            f"时空特征维度错误：{spatio_temporal_features.shape}"
        
        # 3. 特征投影
        features = self.feature_proj(spatio_temporal_features)  # [batch_size, num_players, hidden_dim]
        assert features.size(-1) == self.hidden_dim, \
            f"投影特征维度错误：{features.shape}"
        
        # 4. 领域对抗训练
        if self.training:  # 仅在训练时进行对抗
            domain_output = self.domain_adversarial(
                features,
                lambda_=1.0
            )
            features = domain_output['features']
        
        # 5. 赛季适应
        if season_info is not None:
            season_encoded = self.season_encoder(season_info)
            features = self.seasonal_adaptation(features, season_encoded)
        
        # 6. 融合元知识
        if meta_knowledge is not None:
            features = torch.cat([
                features,
                meta_knowledge.expand(batch_size, -1, -1)
            ], dim=-1)
        
        # 7. 生成策略和价值
        policy_features = self.policy_net(features)  # [batch_size, num_players, 128]
        
        # 对特征进行平均池化，保持维度正确
        # 先检查输入维度
        assert len(features.shape) == 3, f"特征维度错误：{features.shape}"
        assert features.size(-1) == self.hidden_dim, f"特征维度不匹配：{features.shape}"
        
        # 对球员维度进行池化
        pooled_features = features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 确保维度正确
        if len(pooled_features.shape) == 1:
            pooled_features = pooled_features.unsqueeze(0)  # [1, hidden_dim]
        
        # 验证池化后的维度
        assert pooled_features.size(0) == batch_size, \
            f"批次维度错误：{pooled_features.shape} vs {batch_size}"
        assert pooled_features.size(-1) == self.hidden_dim, \
            f"池化特征维度错误：{pooled_features.shape} vs {self.hidden_dim}"
        
        # 生成价值
        state_value = self.value_net(pooled_features)  # [batch_size, 1]
        
        # 8. 生成动作概率
        action_logits = self.action_generator(policy_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 9. 约束处理
        constraint_violations = self._handle_constraints(
            action_probs, state, constraints
        )
        
        return {
            'action_probs': action_probs,
            'state_value': state_value,
            'constraint_violations': constraint_violations,
            'meta_knowledge': meta_knowledge if meta_knowledge is not None else policy_features,
            'domain_features': features,
            'reward': -sum(v.mean() for v in constraint_violations.values())
        }
    
    def _handle_constraints(self, action_probs, state, constraints=None):
        """处理约束"""
        if constraints is not None:
            # 只检查指定的约束
            violations = {}
            for constraint in constraints:
                if constraint == 'budget':
                    violation = self.constraint_checker.check_budget_constraint(
                        action_probs * state['values']
                    )
                    violations['budget'] = violation
                elif constraint == 'position':
                    violation = self.constraint_checker.check_position_constraint(
                        action_probs * state['position_onehot'],
                        torch.tensor([2, 5, 5, 3], device=action_probs.device)
                    )
                    violations['position'] = violation
                # ... 其他约束检查
        else:
            # 检查所有约束
            violations = self.constraint_checker(
                action_probs,
                state
            )
        
        return violations
    
    def compute_loss(self, batch, season_info=None):
        """
        计算损失函数
        Args:
            batch: 训练数据批次
            season_info: 赛季信息
        """
        # 1. 策略和价值损失
        policy_loss = 0
        value_loss = 0
        
        for data in batch:
            output = self.forward(data['state'], season_info=season_info)
            
            # 策略损失
            policy_loss += F.cross_entropy(
                output['action_probs'],
                data['action']
            )
            
            # 价值损失
            value_loss += F.mse_loss(
                output['state_value'],
                data['return']
            )
        
        # 2. 领域对抗损失
        if season_info is not None:
            domain_loss = F.binary_cross_entropy_with_logits(
                output['domain_features'],
                torch.zeros_like(output['domain_features'])  # 目标：无法区分赛季
            )
        else:
            domain_loss = 0
        
        # 3. 约束违反损失
        constraint_loss = sum(
            violation.mean()
            for violation in output['constraint_violations'].values()
        )
        
        # 总损失
        total_loss = (
            policy_loss +
            0.5 * value_loss +
            0.1 * domain_loss +
            0.5 * constraint_loss
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'domain_loss': domain_loss,
            'constraint_loss': constraint_loss
        }
    
    def select_action(self, state, epsilon=0.0):
        """选择动作"""
        with torch.no_grad():
            output = self.forward(state)
            action_probs = output['action_probs']
            
            if torch.rand(1) < epsilon:
                # 探索：随机选择
                action = torch.multinomial(
                    action_probs,
                    num_samples=15,  # 需要选择15名球员
                    replacement=False
                )
            else:
                # 利用：选择概率最高的动作
                _, action = torch.topk(
                    action_probs,
                    k=15,
                    dim=-1
                )
            
            return action
    
    def _build_position_adj_matrix(self, position_onehot):
        """构建基于位置的邻接矩阵"""
        # [batch_size, num_players, num_players]
        return torch.matmul(position_onehot, position_onehot.transpose(-2, -1))
    
    def _build_team_adj_matrix(self, team_onehot):
        """构建基于球队的邻接矩阵"""
        # [batch_size, num_players, num_players]
        return torch.matmul(team_onehot, team_onehot.transpose(-2, -1))
    
    def _build_value_adj_matrix(self, values):
        """构建基于价值的邻接矩阵"""
        # [batch_size, num_players, 1]
        values_expanded = values.unsqueeze(-1)
        # 计算价值差异
        value_diff = torch.abs(values_expanded - values_expanded.transpose(-2, -1))
        # 使用高斯核将差异转换为相似度
        sigma = values.std()
        return torch.exp(-value_diff / (2 * sigma * sigma)) 