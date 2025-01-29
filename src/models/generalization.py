import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np
import logging

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

def evaluate_generalization(model, env, config, season, num_episodes=10):
    """评估模型在特定赛季的泛化性能"""
    try:
        total_rewards = []
        total_points = []
        total_values = []
        position_violations = []
        budget_violations = []
        
        for episode in range(num_episodes):
            state = env.reset(season)
            done = False
            episode_reward = 0
            episode_points = 0
            episode_value = 0
            
            while not done:
                # 获取动作
                with torch.no_grad():
                    action = model.select_action(state)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 记录结果
                episode_reward += reward
                episode_points += info.get('points', 0)
                episode_value += info.get('value', 0)
                
                if not done:
                    state = next_state
            
            total_rewards.append(episode_reward)
            total_points.append(episode_points)
            total_values.append(episode_value)
            position_violations.append(info.get('position_violations', {}))
            budget_violations.append(info.get('budget_violation', 0))
        
        # 计算平均值
        results = {
            'avg_reward': np.mean(total_rewards),
            'avg_points': np.mean(total_points),
            'avg_value': np.mean(total_values),
            'position_violations': position_violations,
            'budget_violations': budget_violations,
            'total_rewards': total_rewards,
            'total_points': total_points,
            'total_values': total_values
        }
        
        return results
        
    except Exception as e:
        print(f"Error in evaluate_generalization: {str(e)}")
        print(f"Season: {season}")
        print(f"Episode: {episode}")
        raise 

class GeneralizationEvaluator:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("FPL-DCIO")
    
    def evaluate_cross_season(self, 
                            model: torch.nn.Module,
                            env,
                            train_seasons: List[str],
                            test_season: str,
                            num_episodes: int = 10) -> Dict:
        """评估跨赛季泛化性能"""
        metrics = {
            'avg_points': [],
            'total_value': [],
            'violations': [],
            'team_diversity': [],
            'synergy_score': []
        }
        
        for _ in range(num_episodes):
            episode_metrics = self._evaluate_single_episode(
                model, env, test_season
            )
            for key in metrics:
                metrics[key].append(episode_metrics[key])
        
        # 计算平均值和标准差
        results = {
            key: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for key, values in metrics.items()
        }
        
        self.logger.info(f"\nCross-season evaluation ({test_season}):")
        for key, value in results.items():
            self.logger.info(
                f"{key}: {value['mean']:.2f} ± {value['std']:.2f}"
            )
        
        return results
    
    def evaluate_covid_impact(self, 
                            model: torch.nn.Module,
                            env,
                            normal_season: str,
                            covid_season: str) -> Dict:
        """评估在COVID赛季的泛化性能"""
        normal_metrics = self.evaluate_cross_season(
            model, env, [normal_season], normal_season
        )
        covid_metrics = self.evaluate_cross_season(
            model, env, [normal_season], covid_season
        )
        
        # 计算性能差异
        impact = {
            key: {
                'relative_change': (
                    covid_metrics[key]['mean'] - normal_metrics[key]['mean']
                ) / normal_metrics[key]['mean'] * 100
            }
            for key in normal_metrics
        }
        
        self.logger.info("\nCOVID Impact Analysis:")
        for key, value in impact.items():
            self.logger.info(
                f"{key} change: {value['relative_change']:.1f}%"
            )
        
        return {
            'normal': normal_metrics,
            'covid': covid_metrics,
            'impact': impact
        }
    
    def evaluate_transfer_window(self, 
                               model: torch.nn.Module,
                               env,
                               season: str,
                               window_gw: int) -> Dict:
        """评估转会窗口前后的性能变化"""
        pre_window = self._evaluate_period(
            model, env, season, 1, window_gw
        )
        post_window = self._evaluate_period(
            model, env, season, window_gw + 1, 38
        )
        
        return {
            'pre_window': pre_window,
            'post_window': post_window
        }
    
    def _evaluate_single_episode(self, model, env, season):
        """评估单个episode的性能"""
        state = env.reset(season=season)
        done = False
        total_points = 0
        violations = 0
        
        while not done:
            action = model.select_action(state, epsilon=0)  # 测试时不需要探索
            next_state, reward, done, info = env.step(action)
            
            total_points += info.get('points', 0)
            violations += info.get('violations', 0)
            
            if not done:
                state = next_state
        
        return {
            'avg_points': total_points / 38,  # 每周平均得分
            'total_value': info.get('total_value', 0),
            'violations': violations,
            'team_diversity': info.get('team_diversity', 0),
            'synergy_score': info.get('synergy_score', 0)
        }
    
    def _evaluate_period(self, model, env, season, start_gw, end_gw):
        """评估特定比赛周期间的性能"""
        # 实现特定期间的评估逻辑
        pass 