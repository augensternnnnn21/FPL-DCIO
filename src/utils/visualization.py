import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, List
import pandas as pd
from pathlib import Path

class VisualizationTool:
    """实验结果可视化工具"""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_tsne_visualization(self, 
                              features_dict: Dict[str, torch.Tensor],
                              title: str = "Feature Distribution"):
        """
        使用T-SNE可视化特征分布
        Args:
            features_dict: 不同赛季的特征字典 {season_name: features}
        """
        plt.figure(figsize=(12, 8))
        
        # 合并所有特征
        all_features = []
        labels = []
        for season_name, features in features_dict.items():
            features = features.cpu().numpy()
            all_features.append(features)
            labels.extend([season_name] * len(features))
        
        all_features = np.concatenate(all_features, axis=0)
        
        # 执行T-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(all_features)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'x': features_2d[:, 0],
            'y': features_2d[:, 1],
            'Season': labels
        })
        
        # 绘制散点图
        sns.scatterplot(data=df, x='x', y='y', hue='Season', alpha=0.6)
        plt.title(title)
        plt.savefig(self.output_dir / f"tsne_{title.lower().replace(' ', '_')}.png")
        plt.close()
    
    def plot_training_curves(self, metrics: dict):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 8))
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.savefig(self.output_dir / 'training_curves.png')
        plt.close()
    
    def plot_generalization_analysis(self, results: Dict):
        """绘制泛化性能分析图"""
        plt.figure(figsize=(15, 12))
        
        # 1. 分布距离热图
        plt.subplot(2, 2, 1)
        dist_matrix = np.array([[results[s1][s2]['distribution_distance'] 
                               for s2 in results] 
                               for s1 in results])
        sns.heatmap(dist_matrix, annot=True, fmt='.3f',
                   xticklabels=results.keys(),
                   yticklabels=results.keys())
        plt.title("Distribution Distance Matrix")
        
        # 2. 适应速度对比
        plt.subplot(2, 2, 2)
        adaptation_rates = [res['adaptation_rate'] for res in results.values()]
        plt.bar(results.keys(), adaptation_rates)
        plt.title("Adaptation Rate by Season")
        plt.xticks(rotation=45)
        
        # 3. 鲁棒性得分
        plt.subplot(2, 2, 3)
        robustness = [res['robustness_score'] for res in results.values()]
        plt.bar(results.keys(), robustness)
        plt.title("Robustness Score by Season")
        plt.xticks(rotation=45)
        
        # 4. 性能随时间变化
        plt.subplot(2, 2, 4)
        for season, res in results.items():
            plt.plot(res['performance_curve'], label=season)
        plt.title("Performance over Time")
        plt.xlabel("Steps")
        plt.ylabel("Performance")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "generalization_analysis.png")
        plt.close()

class ResultAnalyzer:
    """实验结果分析器"""
    def __init__(self):
        self.metrics = {
            'distribution_distance': self._analyze_distribution_distance,
            'robustness': self._analyze_robustness,
            'adaptation': self._analyze_adaptation
        }
    
    def analyze_results(self, results: Dict) -> Dict:
        """分析实验结果"""
        analysis = {}
        for metric_name, analyzer in self.metrics.items():
            analysis[metric_name] = analyzer(results)
        return analysis
    
    def _analyze_distribution_distance(self, results: Dict) -> Dict:
        """分析特征分布距离"""
        distances = results['distribution_distance']
        return {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'max_shift': np.max(distances),
            'min_shift': np.min(distances)
        }
    
    def _analyze_robustness(self, results: Dict) -> Dict:
        """分析模型鲁棒性"""
        scores = results['robustness_score']
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'stability': 1 - np.std(scores) / np.mean(scores)
        }
    
    def _analyze_adaptation(self, results: Dict) -> Dict:
        """分析适应性能"""
        rates = results['adaptation_rate']
        return {
            'mean_rate': np.mean(rates),
            'convergence_speed': 1 / np.mean(rates),
            'adaptation_stability': 1 - np.std(rates) / np.mean(rates)
        }
    
    def generate_report(self, analysis: Dict) -> str:
        """生成分析报告"""
        report = "Experiment Analysis Report\n"
        report += "========================\n\n"
        
        # 1. 分布距离分析
        report += "Feature Distribution Analysis:\n"
        report += f"- Mean distance: {analysis['distribution_distance']['mean']:.4f}\n"
        report += f"- Max shift: {analysis['distribution_distance']['max_shift']:.4f}\n"
        report += f"- Distribution stability: {1 - analysis['distribution_distance']['std']:.4f}\n\n"
        
        # 2. 鲁棒性分析
        report += "Robustness Analysis:\n"
        report += f"- Mean robustness score: {analysis['robustness']['mean_score']:.4f}\n"
        report += f"- Stability index: {analysis['robustness']['stability']:.4f}\n\n"
        
        # 3. 适应性分析
        report += "Adaptation Analysis:\n"
        report += f"- Mean adaptation rate: {analysis['adaptation']['mean_rate']:.4f}\n"
        report += f"- Convergence speed: {analysis['adaptation']['convergence_speed']:.4f}\n"
        report += f"- Adaptation stability: {analysis['adaptation']['adaptation_stability']:.4f}\n"
        
        return report 