import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import mutual_info_score
import torch

class StatisticalAnalyzer:
    """统计分析工具"""
    
    @staticmethod
    def compute_confidence_interval(data: np.ndarray, 
                                 confidence: float = 0.95) -> Tuple[float, float]:
        """计算置信区间"""
        mean = np.mean(data)
        std_err = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=std_err)
        return ci
    
    @staticmethod
    def perform_significance_test(group1: np.ndarray, 
                                group2: np.ndarray,
                                test_type: str = 'ttest') -> Dict:
        """执行显著性检验"""
        if test_type == 'ttest':
            stat, pval = stats.ttest_ind(group1, group2)
            effect_size = np.abs(np.mean(group1) - np.mean(group2)) / np.std(group1)
        elif test_type == 'wilcoxon':
            stat, pval = stats.wilcoxon(group1, group2)
            effect_size = stats.ranksums(group1, group2)[0] / np.sqrt(len(group1))
        
        return {
            'statistic': stat,
            'p_value': pval,
            'effect_size': effect_size,
            'significant': pval < 0.05
        }
    
    @staticmethod
    def analyze_feature_importance(features: torch.Tensor, 
                                 performance: torch.Tensor) -> Dict[str, float]:
        """分析特征重要性"""
        features = features.cpu().numpy()
        performance = performance.cpu().numpy()
        
        importance_scores = {}
        for i in range(features.shape[1]):
            mi = mutual_info_score(features[:, i], performance)
            importance_scores[f'feature_{i}'] = mi
        
        return importance_scores
    
    @staticmethod
    def compute_correlation_matrix(data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """计算相关性矩阵"""
        df = pd.DataFrame(data)
        return df.corr()
    
    @staticmethod
    def analyze_convergence(learning_curves: Dict[str, List[float]]) -> Dict:
        """分析收敛特性"""
        results = {}
        for metric, values in learning_curves.items():
            values = np.array(values)
            
            # 计算收敛速度（使用指数拟合）
            x = np.arange(len(values))
            try:
                from scipy.optimize import curve_fit
                def exp_curve(x, a, b, c):
                    return a * np.exp(-b * x) + c
                
                popt, _ = curve_fit(exp_curve, x, values)
                convergence_rate = popt[1]
            except:
                convergence_rate = None
            
            # 计算稳定性
            stability = 1 - np.std(values[-100:]) / np.mean(values[-100:])
            
            # 计算最终性能
            final_performance = np.mean(values[-100:])
            
            results[metric] = {
                'convergence_rate': convergence_rate,
                'stability': stability,
                'final_performance': final_performance
            }
        
        return results
    
    @staticmethod
    def analyze_generalization_gap(train_perf: np.ndarray, 
                                 test_perf: np.ndarray) -> Dict:
        """分析泛化差距"""
        train_mean = np.mean(train_perf)
        test_mean = np.mean(test_perf)
        gap = train_mean - test_mean
        
        # 计算相对泛化差距
        relative_gap = gap / train_mean
        
        # 执行显著性检验
        significance = StatisticalAnalyzer.perform_significance_test(
            train_perf, test_perf
        )
        
        return {
            'absolute_gap': gap,
            'relative_gap': relative_gap,
            'significance': significance
        }
    
    @staticmethod
    def generate_statistical_report(analysis_results: Dict) -> str:
        """生成统计分析报告"""
        report = "Statistical Analysis Report\n"
        report += "==========================\n\n"
        
        # 1. 性能统计
        report += "Performance Statistics:\n"
        report += "-----------------------\n"
        for metric, stats in analysis_results['performance'].items():
            ci = stats['confidence_interval']
            report += f"{metric}:\n"
            report += f"  Mean: {stats['mean']:.4f}\n"
            report += f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n"
            report += f"  Effect Size: {stats['effect_size']:.4f}\n\n"
        
        # 2. 收敛分析
        report += "Convergence Analysis:\n"
        report += "--------------------\n"
        for metric, conv_stats in analysis_results['convergence'].items():
            report += f"{metric}:\n"
            report += f"  Convergence Rate: {conv_stats['convergence_rate']:.4f}\n"
            report += f"  Stability: {conv_stats['stability']:.4f}\n"
            report += f"  Final Performance: {conv_stats['final_performance']:.4f}\n\n"
        
        # 3. 泛化分析
        report += "Generalization Analysis:\n"
        report += "----------------------\n"
        gen_stats = analysis_results['generalization']
        report += f"Absolute Gap: {gen_stats['absolute_gap']:.4f}\n"
        report += f"Relative Gap: {gen_stats['relative_gap']:.4f}\n"
        report += f"Statistical Significance: {gen_stats['significance']['p_value']:.4f}\n\n"
        
        return report 