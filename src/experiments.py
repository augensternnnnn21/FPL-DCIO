import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from torch import optim
from dataclasses import dataclass, field
from typing import List, Dict, Any

from config import (
    ExperimentConfig, 
    PROJECT_ROOT,
    BUDGET,
    NUM_PLAYERS,
)
from environment import FPLEnv
from models import DynamicSpatioTemporalMetaLearner
from utils.visualization import VisualizationTool, ResultAnalyzer
from utils.statistical_analysis import StatisticalAnalyzer
from models.generalization import evaluate_generalization

from models.dst_meta_grl import DynamicSpatioTemporalMetaLearner



class ExperimentRunner:
    """实验运行器"""
    def __init__(self, config: ExperimentConfig):
        """
        初始化实验运行器
        Args:
            config: 实验配置
        """
        print("\nDEBUG: ExperimentRunner init...")
        
        # 基础配置
        self.config = config
        self.base_data_path = config.data_path  # 使用配置中的data_path
        print(f"Base data path: {self.base_data_path}")
        
        # 确保输出目录存在
        os.makedirs(config.output_path, exist_ok=True)
        
        # 初始化结果存储
        self.results = {
            'training': {},
            'validation': {},
            'testing': {},
            'generalization': {}
        }
        
        # 初始化环境
        print("\nDEBUG: Creating environment...")
        self.current_season = config.train_seasons[0]  # 初始化为第一个训练赛季
        data_path = os.path.join(self.base_data_path, self.current_season, f"{self.current_season}_processed.csv")
        
        # 验证数据文件存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"Full data path: {data_path}")
        print(f"File exists: {os.path.exists(data_path)}")
        
        try:
            # 创建临时配置对象
            env_config = ExperimentConfig(
                data_path=str(data_path),
                budget=BUDGET,
                max_players=NUM_PLAYERS
            )
            print("\nDEBUG: 环境配置对象结构:")
            print(f"类型: {type(env_config)}")
            print(f"包含data_path字段: {'data_path' in env_config.__dict__}")
            print(f"完整配置: {env_config.__dict__}")
            self.env = FPLEnv(env_config)
        except Exception as e:
            print(f"Failed to create environment: {str(e)}")
            raise
        
        # 获取输入维度
        initial_state = self.env.reset()
        input_dim = initial_state.shape[-1] if isinstance(initial_state, np.ndarray) else initial_state['player_features'].shape[-1]
        print(f"\nDetermined input dimension: {input_dim}")
        
        # 创建模型
        print("\nDEBUG: Creating model...")
        self.model = DynamicSpatioTemporalMetaLearner(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim
        ).to(config.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate
        )
        
        # 记录最佳性能
        self.best_reward = float('-inf')
        self.best_model_path = os.path.join(
            config.output_path,
            'best_model.pth'
        )
        
        # 添加可视化和分析工具
        self.visualizer = VisualizationTool(config.output_path)
        self.analyzer = ResultAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def _collect_features(self, seasons):
        """收集特定赛季的特征"""
        features = []
        for season in seasons:
            state, _ = self.env.reset(season=season)
            features.append(self.model.get_representation(state))
        return torch.cat(features, dim=0)
    
    def _get_epsilon(self, step):
        """获取探索率"""
        return max(
            self.config.training.epsilon_end,
            self.config.training.epsilon_start * 
            (self.config.training.epsilon_decay ** step)
        )
    
    def _log_training_progress(self, episode, loss_dict):
        """记录训练进度"""
        print(f"\nEpisode {episode}:")
        for loss_name, loss_value in loss_dict.items():
            print(f"{loss_name}: {loss_value:.4f}")
            self.results['training'].setdefault(loss_name, []).append(loss_value)
    
    def _evaluate_current_model(self, episode):
        """评估当前模型"""
        print(f"\nEpisode {episode}:")
        
        # 打印训练损失
        if hasattr(self, 'latest_loss'):
            for key, value in self.latest_loss.items():
                print(f"{key}: {value:.4f}")
        
        # 在验证集上评估
        try:
            # 重置环境到验证赛季
            state = self.env.reset(season=self.config.val_season)  # 移除对返回值的解包
            
            total_reward = 0
            done = False
            
            while not done:
                # 选择动作
                action_info = self.model.select_action(state)
                action = action_info['action']
                
                # 执行动作
                next_state_info = self.env.step(action)
                next_state = next_state_info['next_state']
                reward = next_state_info['reward']
                done = next_state_info['done']
                
                total_reward += reward
                state = next_state
            
            print(f"Validation reward: {total_reward:.2f}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
    
    def _save_results(self):
        """保存实验结果"""
        save_path = os.path.join(self.config.output_path, 'experiment_results.pkl')
        torch.save(self.results, save_path)
        print(f"\nSaved results to {save_path}")
    
    def run_experiments(self):
        """运行实验"""
        # 保存实验配置
        self._save_config()
        
        # 多次运行以获得统计显著性
        for run in range(self.config.num_runs):
            print(f"\nStarting Run {run + 1}/{self.config.num_runs}")
            
            # 重置随机种子
            torch.manual_seed(self.config.random_seed + run)
            np.random.seed(self.config.random_seed + run)
            
            # 运行单次实验
            self._run_single_experiment()
        
        # 分析多次运行的结果
        self._analyze_multiple_runs()
    
    def _save_config(self):
        """保存实验配置"""
        config_path = os.path.join(self.config.output_path, 'experiment_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4)
    
    def _run_single_experiment(self):
        """运行单次实验"""
        print("\nStarting single experiment...")
        
        # 1. 训练模型
        self._train_model()
        
        # 2. 评估泛化性能
        self._evaluate_generalization()
        
        # 3. 保存和可视化结果
        self._save_results()
        self._visualize_results()
    
    def _train_model(self):
        """训练模型"""
        print("\nTraining model...")
        progress_bar = tqdm(range(self.config.training.num_episodes))
        
        for episode in progress_bar:
            # 随机选择一个训练赛季
            self.current_season = np.random.choice(self.config.train_seasons)
            data_path = os.path.join(self.base_data_path, self.current_season, f"{self.current_season}_processed.csv")
            
            # 创建符合FPLEnv要求的配置对象
            env_config = ExperimentConfig(
                data_path=str(data_path),
                budget=BUDGET,
                max_players=NUM_PLAYERS
            )
            print("\nDEBUG: 环境配置对象结构:")
            print(f"类型: {type(env_config)}")
            print(f"包含data_path字段: {'data_path' in env_config.__dict__}")
            print(f"完整配置: {env_config.__dict__}")
            self.env = FPLEnv(env_config)
            
            # 收集经验数据
            episode_data = self._collect_episode_data(self.current_season)
            
            # 计算损失
            loss_dict = self.model.compute_loss(episode_data, self.current_season)
            
            # 优化器步骤
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # 梯度裁剪
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.grad_clip
                )
            
            self.optimizer.step()
            
            # 记录训练进度
            self._log_training_progress(episode, {
                'policy_loss': loss_dict['policy_loss'].item(),
                'value_loss': loss_dict['value_loss'].item(),
                'total_loss': loss_dict['total_loss'].item()
            })
    
    def _evaluate_generalization(self):
        """评估模型的泛化性能"""
        print("\nEvaluating generalization performance...")
        
        # 1. 常规赛季泛化性
        print("Testing on regular season...")
        regular_results = evaluate_generalization(
            self.model,
            self.config.train_seasons,
            self.config.val_season,
            self.env
        )
        self.results['generalization']['regular'] = regular_results
        
        # 2. COVID赛季泛化性
        print("Testing on COVID season...")
        covid_results = evaluate_generalization(
            self.model,
            self.config.train_seasons,
            self.config.test_season,
            self.env
        )
        self.results['generalization']['covid'] = covid_results
        
        # 打印结果
        self._print_generalization_results()
    
    def _encode_season_info(self, season):
        """编码赛季信息"""
        year = int(season.split('-')[0])
        is_covid = season in ["2019-20", "2020-21"]
        
        # 编码为4维向量：[年份归一化, 是否COVID赛季, sin(年份), cos(年份)]
        year_norm = (year - 2016) / 10  # 归一化年份
        year_rad = 2 * np.pi * year_norm
        
        return torch.tensor([
            year_norm,
            float(is_covid),
            np.sin(year_rad),
            np.cos(year_rad)
        ], device=self.config.device)
    
    def _collect_episode_data(self, season_info):
        """收集单个回合的数据"""
        episode_data = []
        # 修复 reset() 返回值的处理
        initial_state = self.env.reset()
        if isinstance(initial_state, tuple):
            state = initial_state[0]  # 新版本 Gym 返回 (state, info)
        else:
            state = initial_state  # 旧版本直接返回 state
        
        done = False
        
        while not done:
            # 选择动作
            action_info = self.model.select_action(state)
            action = action_info['action']  # [batch_size, num_players]
            
            # 确保动作是布尔张量
            action = action.bool()
            
            # 如果是批处理，只取第一个
            if action.dim() > 1 and action.size(0) > 1:
                action = action[0]
            
            # 打印调试信息
            print(f"\nEpisode data collection:")
            if isinstance(state, dict):
                state_shape = state['player_features'].shape
            else:
                state_shape = state.shape
            print(f"state shape: {state_shape}")
            print(f"action shape: {action.shape}")
            print(f"Selected players: {action.sum().item()}")
            
            # 执行动作
            next_state_info = self.env.step(action)
            
            # 处理返回值
            if len(next_state_info) == 5:  # 新版本 Gym 返回 (next_state, reward, done, truncated, info)
                next_state, reward, done, _, _ = next_state_info
            else:  # 旧版本返回 (next_state, reward, done, info)
                next_state, reward, done, _ = next_state_info
            
            # 存储转换
            episode_data.append({
                'state': state,
                'action': action,
                'log_prob': action_info['log_prob'],
                'value': action_info['value'],
                'reward': reward,
                'violations': action_info.get('violations', None),  # 使用 get 方法避免 KeyError
                'next_state': next_state,
                'done': done,
                'season_info': season_info
            })
            
            state = next_state
        
        return episode_data
    
    def _visualize_results(self):
        """可视化实验结果"""
        # 1. T-SNE可视化
        features_dict = {
            'Train': self.collect_features(self.config.train_seasons),
            'Validation': self.collect_features([self.config.val_season]),
            'Test': self.collect_features([self.config.test_season])
        }
        self.visualizer.plot_tsne_visualization(features_dict)
        
        # 2. 训练曲线
        self.visualizer.plot_training_curves(self.results['training'])
        
        # 3. 泛化性能分析
        self.visualizer.plot_generalization_analysis(self.results['generalization'])
    
    def _print_generalization_results(self):
        """打印泛化性能结果"""
        print("\nGeneralization Results:")
        print("\nRegular Season (2021-22):")
        for metric, value in self.results['generalization']['regular'].items():
            print(f"{metric}: {value:.4f}")
        
        print("\nCOVID Season (2022-23):")
        for metric, value in self.results['generalization']['covid'].items():
            print(f"{metric}: {value:.4f}")

    def _analyze_multiple_runs(self):
        """分析多次运行的结果"""
        # 1. 收集所有运行的数据
        all_runs_data = {
            'train_performance': [],
            'test_performance': [],
            'features': [],
            'learning_curves': defaultdict(list)
        }
        
        for run_results in self.results['runs']:
            all_runs_data['train_performance'].append(run_results['train_performance'])
            all_runs_data['test_performance'].append(run_results['test_performance'])
            all_runs_data['features'].append(run_results['features'])
            
            for metric, values in run_results['learning_curves'].items():
                all_runs_data['learning_curves'][metric].append(values)
        
        # 2. 执行统计分析
        analysis_results = {}
        
        # 2.1 性能统计
        train_perf = np.array(all_runs_data['train_performance'])
        test_perf = np.array(all_runs_data['test_performance'])
        
        analysis_results['performance'] = {
            'train': {
                'mean': np.mean(train_perf),
                'confidence_interval': self.statistical_analyzer.compute_confidence_interval(train_perf),
                'effect_size': np.std(train_perf)
            },
            'test': {
                'mean': np.mean(test_perf),
                'confidence_interval': self.statistical_analyzer.compute_confidence_interval(test_perf),
                'effect_size': np.std(test_perf)
            }
        }
        
        # 2.2 收敛分析
        analysis_results['convergence'] = self.statistical_analyzer.analyze_convergence(
            {k: np.mean(v, axis=0) for k, v in all_runs_data['learning_curves'].items()}
        )
        
        # 2.3 泛化分析
        analysis_results['generalization'] = self.statistical_analyzer.analyze_generalization_gap(
            train_perf, test_perf
        )
        
        # 2.4 特征重要性分析
        features = torch.cat(all_runs_data['features'])
        performance = torch.cat([
            torch.tensor(train_perf),
            torch.tensor(test_perf)
        ])
        analysis_results['feature_importance'] = self.statistical_analyzer.analyze_feature_importance(
            features, performance
        )
        
        # 3. 生成报告
        report = self.statistical_analyzer.generate_statistical_report(analysis_results)
        
        # 4. 保存结果
        with open(os.path.join(self.config.output_path, 'statistical_analysis.txt'), 'w') as f:
            f.write(report)
        
        return analysis_results

def main():
    # 使用不同的配置运行实验
    configs = {
        'default': DEFAULT_CONFIG,
        'covid': COVID_CONFIG
    }
    
    for name, config in configs.items():
        print(f"\nRunning {name} experiments...")
        try:
            runner = ExperimentRunner(config)
            runner.run_experiments()
        except Exception as e:
            print(f"Error running {name} experiments: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 