import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np

class DomainAdversarialTraining(nn.Module):
    """领域对抗训练模块，用于增强跨赛季泛化性"""
    def __init__(self, feature_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, feature_dim)  # 确保输出维度与输入相同
        )
        
        # 领域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 1)  # 二分类：源域vs目标域
        )
        
        # 梯度反转层的alpha参数
        self.alpha = 1.0
    
    def forward(self, features, lambda_=1.0):
        """
        前向传播，包含梯度反转
        Args:
            features: [batch_size, num_players, feature_dim] 或 [batch_size, feature_dim]
            lambda_: 梯度反转强度
        """
        # 保存原始维度信息
        original_shape = features.shape
        batch_size = features.size(0)
        
        # 如果是三维输入，先展平
        if len(features.shape) == 3:
            num_players = features.size(1)
            features = features.view(batch_size * num_players, -1)
        
        # 验证输入维度
        assert features.size(-1) == self.feature_dim, \
            f"特征维度不匹配：{features.shape} vs {self.feature_dim}"
        
        # 提取特征
        extracted = self.feature_extractor(features)
        
        # 梯度反转
        reversed_features = GradientReversal.apply(extracted, lambda_)
        
        # 领域分类
        domain_pred = self.domain_classifier(reversed_features)
        
        # 恢复原始维度
        if len(original_shape) == 3:
            extracted = extracted.view(batch_size, num_players, -1)
        
        # 验证输出维度
        assert extracted.size(-1) == self.feature_dim, \
            f"输出维度错误：{extracted.shape} vs {self.feature_dim}"
        
        return {
            'features': extracted,
            'domain_pred': domain_pred
        }

class GradientReversal(torch.autograd.Function):
    """梯度反转层"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class SeasonalAdaptationNetwork(nn.Module):
    """赛季适应网络"""
    def __init__(self, input_dim=64):
        super().__init__()
        
        # 赛季特征编码器
        self.season_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 16)
        )
        
        # 适应层
        self.adaptation_layer = nn.Sequential(
            nn.Linear(input_dim + 16, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim)
        )
    
    def forward(self, features, season_info):
        """
        前向传播
        Args:
            features: 球员特征
            season_info: 赛季信息
        """
        # 编码赛季特征
        season_features = self.season_encoder(season_info)
        
        # 扩展赛季特征以匹配球员特征维度
        season_features = season_features.unsqueeze(1).expand(-1, features.size(1), -1)
        
        # 特征融合
        combined = torch.cat([features, season_features], dim=-1)
        
        # 适应
        adapted = self.adaptation_layer(combined)
        
        return adapted

class GeneralizationMetrics:
    """泛化性能评估指标"""
    @staticmethod
    def compute_distribution_distance(source_features: torch.Tensor, 
                                   target_features: torch.Tensor) -> float:
        """计算特征分布距离"""
        # 使用MMD（Maximum Mean Discrepancy）
        def gaussian_kernel(x, y, sigma=1.0):
            dist = torch.cdist(x, y)
            return torch.exp(-dist / (2 * sigma * sigma))
        
        xx = gaussian_kernel(source_features, source_features)
        yy = gaussian_kernel(target_features, target_features)
        xy = gaussian_kernel(source_features, target_features)
        
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd.item()
    
    @staticmethod
    def compute_robustness_score(model,
                               states: List[Dict],
                               perturbation_ratio: float = 0.1) -> float:
        """计算模型鲁棒性得分"""
        original_outputs = []
        perturbed_outputs = []
        
        with torch.no_grad():
            for state in states:
                # 原始预测
                orig_output = model(state)
                original_outputs.append(orig_output['action_probs'])
                
                # 添加扰动
                perturbed_state = {
                    k: v + torch.randn_like(v) * perturbation_ratio 
                    if torch.is_tensor(v) else v
                    for k, v in state.items()
                }
                pert_output = model(perturbed_state)
                perturbed_outputs.append(pert_output['action_probs'])
        
        # 计算预测稳定性
        original_tensor = torch.stack(original_outputs)
        perturbed_tensor = torch.stack(perturbed_outputs)
        stability = F.kl_div(
            perturbed_tensor.log(),
            original_tensor,
            reduction='batchmean'
        )
        
        return -stability.item()  # 负的KL散度，越大越稳定
    
    @staticmethod
    def compute_adaptation_speed(model,
                               source_data: List[Dict],
                               target_data: List[Dict],
                               num_steps: int = 100) -> float:
        """计算模型适应速度"""
        performance_curve = []
        
        # 在源域上的初始性能
        source_perf = model.evaluate(source_data)
        
        # 在目标域上逐步适应
        for step in range(num_steps):
            target_perf = model.evaluate(target_data)
            performance_curve.append(target_perf)
        
        # 计算适应速度（使用性能提升的指数拟合）
        x = np.arange(num_steps)
        y = np.array(performance_curve)
        
        try:
            # 拟合指数曲线 y = a * (1 - exp(-bx))
            from scipy.optimize import curve_fit
            def exp_curve(x, a, b):
                return a * (1 - np.exp(-b * x))
            
            popt, _ = curve_fit(exp_curve, x, y)
            adaptation_rate = popt[1]  # b参数表示适应速度
        except:
            # 如果拟合失败，使用简单的性能提升速度
            adaptation_rate = (performance_curve[-1] - performance_curve[0]) / num_steps
        
        return adaptation_rate

def evaluate_generalization(model, train_seasons, test_season, env):
    """评估模型的泛化性能"""
    print("\nEvaluating generalization performance...")
    
    # 收集训练赛季的表现
    train_performance = []
    for season in train_seasons:
        # 重置环境到指定赛季
        state = env.reset(season=season)  # 移除对返回值的解包
        
        total_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action_info = model.select_action(state)
            action = action_info['action']
            
            # 执行动作
            next_state_info = env.step(action)
            if isinstance(next_state_info, tuple):
                if len(next_state_info) == 5:  # 新版本 Gym 返回 (next_state, reward, done, truncated, info)
                    next_state, reward, done, _, _ = next_state_info
                else:  # 旧版本返回 (next_state, reward, done, info)
                    next_state, reward, done, _ = next_state_info
            else:  # 字典格式
                next_state = next_state_info['next_state']
                reward = next_state_info['reward']
                done = next_state_info['done']
            
            total_reward += reward
            state = next_state
        
        train_performance.append(total_reward)
    
    # 在测试赛季上评估
    state = env.reset(season=test_season)  # 移除对返回值的解包
    
    total_reward = 0
    done = False
    
    while not done:
        # 选择动作
        action_info = model.select_action(state)
        action = action_info['action']
        
        # 执行动作
        next_state_info = env.step(action)
        if isinstance(next_state_info, tuple):
            if len(next_state_info) == 5:  # 新版本 Gym 返回 (next_state, reward, done, truncated, info)
                next_state, reward, done, _, _ = next_state_info
            else:  # 旧版本返回 (next_state, reward, done, info)
                next_state, reward, done, _ = next_state_info
        else:  # 字典格式
            next_state = next_state_info['next_state']
            reward = next_state_info['reward']
            done = next_state_info['done']
        
        total_reward += reward
        state = next_state
    
    # 计算泛化性指标
    train_mean = np.mean(train_performance)
    train_std = np.std(train_performance)
    generalization_gap = train_mean - total_reward
    
    return {
        'train_performance': {
            'mean': train_mean,
            'std': train_std,
            'raw': train_performance
        },
        'test_performance': total_reward,
        'generalization_gap': generalization_gap
    } 