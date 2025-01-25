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

from environment import FPLEnv
from models import DynamicSpatioTemporalMetaLearner
from config import *
from utils.visualization import VisualizationTool, ResultAnalyzer
from utils.statistical_analysis import StatisticalAnalyzer
from models.generalization import evaluate_generalization

from models.dst_meta_grl import DynamicSpatioTemporalMetaLearner



class ExperimentRunner:
    """实验运行器"""
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.save_dir = os.path.join(PROJECT_ROOT, "experiments")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置随机种子
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        # 初始化环境和模型
        self.env = FPLEnv(season=self.config.train_seasons[0])
        self.model = DynamicSpatioTemporalMetaLearner(
            input_dim=self.config.model.input_dim,
            hidden_dim=self.config.model.hidden_dim
        ).to(self.config.device)
        
        # 初始化结果存储
        self.results = defaultdict(dict)
        self.results['runs'] = []  # 添加runs列表来存储多次运行的结果
        
        # 添加可视化和分析工具
        self.visualizer = VisualizationTool(self.save_dir)
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
        self.model.eval()
        with torch.no_grad():
            state, _ = self.env.reset(season=self.config.val_season)
            total_reward = 0
            done = False
            
            while not done:
                action = self.model.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            
            print(f"Evaluation at episode {episode}: {total_reward:.2f}")
            self.results['evaluation'].setdefault('rewards', []).append(total_reward)
    
    def _save_results(self):
        """保存实验结果"""
        save_path = os.path.join(self.save_dir, 'experiment_results.pkl')
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
        config_path = os.path.join(self.save_dir, 'experiment_config.json')
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
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for episode in tqdm(range(self.config.training.num_episodes)):
            # 随机选择一个训练赛季
            train_season = np.random.choice(self.config.train_seasons)
            self.env.reset(season=train_season)
            
            # 准备赛季信息
            season_info = self._encode_season_info(train_season)
            
            # 收集训练数据
            episode_data = self._collect_episode_data(season_info)
            
            # 更新模型
            self.model.train()
            loss_dict = self.model.compute_loss(episode_data, season_info)
            
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            # 记录结果
            if episode % self.config.log_interval == 0:
                self._log_training_progress(episode, loss_dict)
            
            # 定期评估
            if episode % self.config.evaluate_interval == 0:
                self._evaluate_current_model(episode)
    
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
        """收集单个episode的训练数据"""
        episode_data = []
        state, _ = self.env.reset()
        done = False
        
        while not done:
            # 选择动作
            action = self.model.select_action(
                state,
                epsilon=self._get_epsilon(len(episode_data))
            )
            
            # 执行动作
            next_state, reward, done, _, _ = self.env.step(action)
            
            # 存储转换
            episode_data.append({
                'state': state,
                'action': action,
                'reward': reward,
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
        with open(os.path.join(self.save_dir, 'statistical_analysis.txt'), 'w') as f:
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