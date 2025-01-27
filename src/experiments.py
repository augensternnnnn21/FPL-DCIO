import os
import json
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
import time

from config import (
    ExperimentConfig, 
    PROJECT_ROOT,
    BUDGET,
    NUM_PLAYERS,
    DEFAULT_CONFIG,
    COVID_CONFIG,
    ModelConfig
)
from environment import FPLEnv
from models.st_conv import DilatedTCN
from utils.logger import FPLLogger
from utils.visualization import VisualizationTool, ResultAnalyzer
from utils.statistical_analysis import StatisticalAnalyzer
from models.generalization import evaluate_generalization, GeneralizationEvaluator
from utils.replay_buffer import ReplayBuffer
from train import train_model

class ExperimentRunner:
    """实验运行器"""
    def __init__(self, config: ExperimentConfig):
        """
        初始化实验运行器
        Args:
            config: 实验配置
        """
        # 设置设备
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化logger
        self.logger = FPLLogger(os.path.join(config.output_path, 'logs'))
        
        # 保存基础配置
        self.config = config
        self.base_data_path = Path(config.data_path)
        
        # 创建环境
        try:
            self.env = FPLEnv(config)
        except Exception as e:
            self.logger.error(f"Failed to create environment: {str(e)}")
            raise
        
        # 获取初始状态以确定输入维度
        initial_state = self.env.reset()
        input_dim = initial_state['player_features'].shape[1]
        self.logger.info(f"Input dimension: {input_dim}")
        
        # 创建模型
        self.logger.info("Creating model...")
        try:
            self.model = self._create_model(
                input_dim=input_dim,
                model_config=self.config.model
            )
            self.target_model = self._create_model(
                input_dim=input_dim,
                model_config=self.config.model
            )
            self.target_model.load_state_dict(self.model.state_dict())
        except Exception as e:
            self.logger.error(f"Failed to create model: {str(e)}")
            raise
        
        # 创建优化器和经验回放
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(config.memory_size)
        
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
        
        # 初始化训练相关的属性
        self.episode_rewards = []
        self.episode_losses = []
        self.current_season = None
        
        # 添加探索率相关属性
        self.initial_epsilon = 0.5
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995
        
        # 创建输出目录
        os.makedirs(config.output_path, exist_ok=True)
        
        # 保存实验配置
        self._save_config()
        
        # 允许用户灵活指定是否使用ILP和Synergy
        self.use_ilp = config.use_ilp if hasattr(config, 'use_ilp') else False
        self.use_synergy = config.use_synergy if hasattr(config, 'use_synergy') else False
        
        if config.use_triple_captain:
            self.env.special_cards['triple_captain'] = True
        if config.use_bench_boost:
            self.env.special_cards['bench_boost'] = True
        
        # 初始化结果字典
        self.results = {
            'training': {},
            'validation': {},
            'testing': {},
            'generalization': {
                'regular': {},
                'covid': {}
            }
        }
    
    def _create_model(self, input_dim: int, model_config: ModelConfig):
        """创建模型"""
        try:
            self.logger.info(f"Creating model with input_dim={input_dim}")
            model = DilatedTCN(
                input_dim=input_dim,
                hidden_dim=model_config.hidden_dim,
                n_layers=model_config.num_layers,
                dropout=model_config.dropout,
                bidirectional=model_config.bidirectional,
                use_attention=model_config.use_attention,
                device=self.device,
                budget=self.config.budget,
                use_ilp=self.config.use_ilp,
                use_synergy=self.config.use_synergy,
                max_players=self.config.max_players
            )
            return model
        
        except Exception as e:
            self.logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def _collect_features(self, seasons):
        """收集特定赛季的特征"""
        features = []
        for season in seasons:
            state, _ = self.env.reset(season=season)
            features.append(self.model.get_representation(state))
        return torch.cat(features, dim=0)
    
    def _get_epsilon(self, step):
        """获取探索率"""
        epsilon_end = 0.01    # 设置默认值
        epsilon_start = 1.0   # 设置默认值
        epsilon_decay = 0.995 # 设置默认值
        
        return max(
            epsilon_end,
            epsilon_start * (epsilon_decay ** step)
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
    
    def _create_env(self):
        """创建环境"""
        try:
            # 验证数据路径
            if not os.path.exists(self.config.data_path):
                raise ValueError(f"Data path not found: {self.config.data_path}")
            
            # 创建环境配置
            env_config = {
                'config': self.config  # 传递整个配置对象
            }
            
            # 创建环境
            env = FPLEnv(**env_config)
            
            # 验证环境
            self.logger.info("Validating environment...")
            initial_state = env.reset()
            
            # 打印环境信息
            self.logger.info(f"Action space: {env.action_space}")
            self.logger.info(f"Observation space: {env.observation_space}")
            
            # 打印状态空间的形状
            for key, value in initial_state.items():
                if hasattr(value, 'shape'):
                    self.logger.info(f"{key} shape: {value.shape}")
            
            return env
            
        except Exception as e:
            self.logger.error(f"Error creating environment: {str(e)}")
            raise

    def run_experiments(self):
        """运行实验"""
        try:
            # 创建环境
            env = self._create_env()
            
            # 临时禁用ILP（因为CBC库兼容性问题）
            self.config.use_ilp = False
            
            # 获取输入维度
            initial_state = env.reset()
            input_dim = initial_state['player_features'].shape[1]  # 使用player_features的特征维度
            self.logger.info(f"Input dimension: {input_dim}")
            
            # 创建模型
            model = self._create_model(
                input_dim=input_dim,
                model_config=self.config.model
            )
            
            # 创建日志记录器
            logger = FPLLogger(os.path.join(self.config.output_path, 'train_logs'))
            
            print("\nRunning baseline experiment...")
            
            # 训练模型
            train_model(
                env=env,
                model=model,
                config=self.config,
                logger=logger
            )
            
            # 保存结果
            self._save_results()
            
        except Exception as e:
            print(f"Error running experiments: {str(e)}")
            raise
    
    def _save_config(self):
        """保存实验配置"""
        config_path = os.path.join(self.config.output_path, 'experiment_config.json')
        
        # 将配置转换为可序列化的字典
        config_dict = {
            'data_path': str(self.config.data_path),
            'train_seasons': self.config.train_seasons,
            'val_season': self.config.val_season,
            'test_season': self.config.test_season,
            'budget': self.config.budget,
            'max_players': self.config.max_players,
            'num_episodes': self.config.num_episodes,
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'output_path': str(self.config.output_path),
            'model': {
                'hidden_dim': self.config.model.hidden_dim,
                'num_layers': self.config.model.num_layers,
                'dropout': self.config.model.dropout,
                'bidirectional': self.config.model.bidirectional,
                'use_attention': self.config.model.use_attention
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
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
        
        return {
            'train_performance': self.best_reward,
            'test_performance': self.best_reward,
            'features': self._collect_features(self.config.train_seasons + [self.config.val_season, self.config.test_season]),
            'learning_curves': {
                'train': self.results['training'],
                'validation': self.results['validation'],
                'testing': self.results['testing']
            }
        }
    
    def _train_model(self):
        """训练模型"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Starting Model Training")
        self.logger.info("="*50 + "\n")
        
        for episode in range(self.config.num_episodes):
            try:
                # 随机选择一个训练赛季
                season = np.random.choice(self.config.train_seasons)
                
                # 收集数据
                episode_data = self._collect_episode_data(season)
                
                # 更新模型
                loss_dict = self.model.update(episode_data)
                self.latest_loss = loss_dict
                
                # 记录奖励
                self.episode_rewards.append(episode_data['total_reward'])
                
                # 每10轮评估一次
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    self.logger.info(f"\nEpisode {episode+1}/{self.config.num_episodes}")
                    self.logger.info(f"Average Reward (last 10): {avg_reward:.2f}")
                    self.logger.info(f"Latest Loss: {loss_dict}")
                    self._evaluate_current_model(episode)
                    
            except Exception as e:
                self.logger.error(f"Error in episode {episode}: {str(e)}")
                continue
    
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
        ], device=self.device)
    
    def _collect_episode_data(self, season):
        """收集一个回合的数据"""
        try:
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            # 重置环境
            state = self.env.reset(season)
            done = False
            total_reward = 0
            steps = 0
            violations = 0
            
            # 获取当前epsilon值
            current_episode = len(self.episode_rewards)
            epsilon = max(self.min_epsilon, 
                         self.initial_epsilon * (self.epsilon_decay ** current_episode))
            
            while not done:
                # 处理状态
                if isinstance(state, dict):
                    # 连接所有特征
                    features = [
                        state['player_features'],
                        state['position_onehot'],
                        state['team_onehot']
                    ]
                    
                    # 确保所有特征都是2D的 [num_players, features]
                    features = [f.squeeze() if len(f.shape) > 2 else f for f in features]
                    
                    # 连接特征
                    state_tensor = np.concatenate(features, axis=1)
                    state_tensor = torch.FloatTensor(state_tensor)
                    
                    # 添加batch维度 [1, num_players, total_features]
                    state_tensor = state_tensor.unsqueeze(0)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                state_tensor = state_tensor.to(self.device)
                
                # 获取动作
                with torch.no_grad():
                    action = self.model.select_action(state_tensor, epsilon=epsilon)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action.cpu().numpy())
                total_reward += reward
                steps += 1
                violations += info.get('violations', 0)
                
                # 存储转换（保持原始numpy格式）
                states.append(state_tensor.cpu().numpy())
                actions.append(action.cpu().numpy())
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                if not done:
                    state = next_state
            
            # 记录训练信息（减少日志频率）
            if current_episode % 10 == 0:  # 每10轮记录一次
                metrics = {
                    'reward': total_reward,
                    'loss': self.latest_loss.get('total_loss', 0),
                    'violations': violations,
                    'team_value': info.get('total_cost', 0),
                    'points': info.get('points', 0),
                    'steps': steps,
                    'epsilon': epsilon,
                    'season': season
                }
                
                if hasattr(self, 'latest_loss'):
                    metrics.update({
                        'policy_loss': self.latest_loss.get('policy_loss', 0),
                        'value_loss': self.latest_loss.get('value_loss', 0)
                    })
                
                self.logger.log_training(current_episode, metrics)
            
            return {
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'next_states': next_states,
                'dones': np.array(dones),
                'total_reward': total_reward,
                'info': info
            }
        
        except Exception as e:
            self.logger.error(f"Error in _collect_episode_data: {str(e)}")
            if isinstance(state, dict):
                self.logger.error(f"State format: {type(state)}")
                self.logger.error(f"State keys: {state.keys()}")
                for k, v in state.items():
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        self.logger.error(f"{k} shape: {v.shape}")
                        if isinstance(v, torch.Tensor):
                            self.logger.error(f"{k} device: {v.device}")
            raise
    
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

    def run(self):
        """运行实验的主入口点"""
        print("\nStarting experiment...")
        
        try:
            # 1. 训练模型
            print("\n[1/3] Training model...")
            self.run_experiments()
            
            # 2. 评估泛化性能
            print("\n[2/3] Evaluating generalization...")
            self._evaluate_generalization()
            
            # 3. 保存和可视化结果
            print("\n[3/3] Saving and visualizing results...")
            self._save_results()
            self._visualize_results()
            
            print("\nExperiment completed successfully!")
            
        except Exception as e:
            print(f"\nError during experiment: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

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