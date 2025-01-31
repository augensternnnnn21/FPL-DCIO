from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

@dataclass
class PredictionMetrics:
    """预测指标"""
    accuracy: float
    roi: float  # Return on Investment
    hit_rate: float  # 命中率
    timing_accuracy: float  # 时机准确度
    confidence_calibration: float  # 置信度校准

class ModelEvaluator:
    """模型评估器"""
    def __init__(self):
        self.metrics_history: List[PredictionMetrics] = []
        
    def evaluate_predictions(self,
                           predictions: List[Tuple[int, int, float]],  # (old_id, new_id, confidence)
                           actual_points: Dict[int, int],  # 实际得分
                           transfer_timing: Dict[int, datetime],  # 实际最佳转会时机
                           budget_used: float) -> PredictionMetrics:
        """评估预测结果"""
        
        # 计算准确率 (预测的球员是否真的表现更好)
        correct_predictions = sum(
            1 for old_id, new_id, _ in predictions
            if actual_points[new_id] > actual_points[old_id]
        )
        accuracy = correct_predictions / len(predictions)
        
        # 计算ROI
        points_gained = sum(
            actual_points[new_id] - actual_points[old_id]
            for old_id, new_id, _ in predictions
        )
        roi = points_gained / budget_used if budget_used > 0 else 0
        
        # 计算命中率 (预测分数与实际分数的接近程度)
        hit_rate = self._calculate_hit_rate(predictions, actual_points)
        
        # 计算时机准确度
        timing_accuracy = self._calculate_timing_accuracy(predictions, transfer_timing)
        
        # 计算置信度校准
        confidence_calibration = self._calculate_confidence_calibration(
            predictions, actual_points
        )
        
        metrics = PredictionMetrics(
            accuracy=accuracy,
            roi=roi,
            hit_rate=hit_rate,
            timing_accuracy=timing_accuracy,
            confidence_calibration=confidence_calibration
        )
        
        self.metrics_history.append(metrics)
        return metrics
        
    def _calculate_hit_rate(self,
                           predictions: List[Tuple[int, int, float]],
                           actual_points: Dict[int, int],
                           threshold: float = 4) -> float:
        """计算命中率"""
        hits = 0
        for old_id, new_id, _ in predictions:
            point_diff = actual_points[new_id] - actual_points[old_id]
            if abs(point_diff) <= threshold:
                hits += 1
        return hits / len(predictions)
        
    def _calculate_timing_accuracy(self,
                                 predictions: List[Tuple[int, int, float]],
                                 actual_timing: Dict[int, datetime]) -> float:
        """计算时机准确度"""
        timing_errors = []
        for _, new_id, _ in predictions:
            if new_id in actual_timing:
                # 计算预测时机与实际最佳时机的时间差
                time_diff = abs((actual_timing[new_id] - datetime.now()).days)
                timing_errors.append(time_diff)
        
        return 1 / (1 + np.mean(timing_errors)) if timing_errors else 0
        
    def _calculate_confidence_calibration(self,
                                        predictions: List[Tuple[int, int, float]],
                                        actual_points: Dict[int, int]) -> float:
        """计算置信度校准"""
        confidences = []
        outcomes = []
        
        for old_id, new_id, confidence in predictions:
            confidences.append(confidence)
            outcomes.append(1 if actual_points[new_id] > actual_points[old_id] else 0)
            
        return np.abs(np.mean(confidences) - np.mean(outcomes))

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算评估指标，使用改进的评估方法
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        Dict[str, float]: 评估指标字典
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 检查维度是否匹配
    if y_true.shape != y_pred.shape:
        raise ValueError(f"维度不匹配: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    
    # 数据预处理
    def preprocess_data(x):
        """预处理数据，处理异常值和标准化"""
        # Winsorization
        p1, p99 = np.percentile(x, [1, 99])
        x = np.clip(x, p1, p99)
        # Min-Max标准化
        x_min, x_max = np.min(x), np.max(x)
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        return x
    
    # 预处理数据
    y_true_processed = preprocess_data(y_true)
    y_pred_processed = preprocess_data(y_pred)
    
    # 1. 基础误差指标
    mse = mean_squared_error(y_true_processed, y_pred_processed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_processed, y_pred_processed)
    
    # 2. 改进的R2计算
    def improved_r2(y_true, y_pred):
        """计算改进的R2分数"""
        residual_ss = np.sum((y_true - y_pred) ** 2)
        total_ss = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - residual_ss / (total_ss + 1e-8)
        # 使用sigmoid函数将R2映射到[0,1]区间
        return 1 / (1 + np.exp(-r2))
    
    r2 = improved_r2(y_true_processed, y_pred_processed)
    
    # 3. 改进的MAPE计算
    def improved_mape(y_true, y_pred):
        """计算改进的MAPE，避免除零问题"""
        # 使用平均值作为基准
        base = np.mean(np.abs(y_true)) + 1e-8
        # 计算相对误差
        relative_errors = np.abs(y_true - y_pred) / base
        # 限制最大相对误差
        relative_errors = np.minimum(relative_errors, 1.0)
        return np.mean(relative_errors) * 100
    
    mape = improved_mape(y_true, y_pred)
    
    # 4. 相关性指标
    def correlation_metrics(y_true, y_pred):
        """计算多个相关性指标"""
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
        # Spearman相关系数
        spearman = stats.spearmanr(y_true, y_pred)[0]
        # Kendall's Tau
        kendall = stats.kendalltau(y_true, y_pred)[0]
        return {
            'Pearson': pearson,
            'Spearman': spearman,
            'Kendall': kendall
        }
    
    correlations = correlation_metrics(y_true_processed, y_pred_processed)
    
    # 5. 多层次准确率计算
    def calculate_accuracies(y_true, y_pred):
        """计算多个层次的准确率"""
        std = np.std(y_true)
        thresholds = {
            'strict': std * 0.25,
            'moderate': std * 0.5,
            'loose': std * 1.0
        }
        accuracies = {}
        for name, threshold in thresholds.items():
            accurate = np.abs(y_true - y_pred) <= threshold
            accuracies[name] = np.mean(accurate) * 100
        return accuracies
    
    accuracies = calculate_accuracies(y_true_processed, y_pred_processed)
    
    # 6. 趋势预测准确率
    def calculate_trend_metrics(y_true, y_pred):
        """计算趋势预测相关指标"""
        # 方向准确率
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        direction_accuracy = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100
        
        # 趋势强度相关性
        trend_correlation = np.corrcoef(np.abs(true_diff), np.abs(pred_diff))[0, 1]
        
        return {
            'Direction': direction_accuracy,
            'Strength': trend_correlation
        }
    
    trend_metrics = calculate_trend_metrics(y_true_processed, y_pred_processed)
    
    # 7. 分位数准确率
    def calculate_quantile_metrics(y_true, y_pred):
        """计算分位数相关指标"""
        quantiles = np.percentile(y_true, [25, 50, 75])
        true_quartiles = np.digitize(y_true, quantiles)
        pred_quartiles = np.digitize(y_pred, quantiles)
        
        # 整体准确率
        accuracy = np.mean(true_quartiles == pred_quartiles) * 100
        
        # 每个分位数的准确率
        quartile_accuracies = {}
        for i in range(1, 5):
            true_mask = true_quartiles == i
            if np.any(true_mask):
                quartile_accuracies[f'Q{i}'] = np.mean(true_quartiles[true_mask] == pred_quartiles[true_mask]) * 100
        
        return {
            'Overall': accuracy,
            'Quartiles': quartile_accuracies
        }
    
    quantile_metrics = calculate_quantile_metrics(y_true_processed, y_pred_processed)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Correlations': correlations,
        'Accuracy_Strict': accuracies['strict'],
        'Accuracy_Moderate': accuracies['moderate'],
        'Accuracy_Loose': accuracies['loose'],
        'Trend_Direction_Accuracy': trend_metrics['Direction'],
        'Trend_Strength': trend_metrics['Strength'],
        'Quantile_Accuracy': quantile_metrics['Overall'],
        'Quantile_Details': quantile_metrics['Quartiles']
    } 