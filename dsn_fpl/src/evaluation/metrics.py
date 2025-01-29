from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime

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