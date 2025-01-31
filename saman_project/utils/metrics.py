import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch

def compute_metrics(predictions, targets, average='weighted'):
    """计算各种评估指标
    
    Args:
        predictions: 模型预测结果
        targets: 真实标签
        average: 多分类指标的平均方式
        
    Returns:
        包含各种指标的字典
    """
    # 确保输入是numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # 基础指标
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average=average
    )
    
    # 混淆矩阵
    cm = confusion_matrix(targets, predictions)
    
    # 计算每个类别的准确率
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # 如果是二分类问题，计算AUC
    if len(np.unique(targets)) == 2:
        auc = roc_auc_score(targets, predictions)
    else:
        auc = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_accuracy,
        'auc': auc
    }

def compute_strategy_metrics(model_outputs, targets):
    """计算策略相关的特殊指标
    
    Args:
        model_outputs: 模型输出字典
        targets: 真实标签
        
    Returns:
        策略相关指标的字典
    """
    metrics = {}
    
    # 策略预测准确性
    predictions = torch.argmax(model_outputs['final_strategy_logits'], dim=1)
    metrics['strategy_accuracy'] = accuracy_score(
        targets.cpu().numpy(),
        predictions.cpu().numpy()
    )
    
    # 策略一致性分数
    strategy_consistency = torch.mean(
        torch.cosine_similarity(
            model_outputs['strategy_logits'],
            model_outputs['meta_strategy_logits']
        )
    ).item()
    metrics['strategy_consistency'] = strategy_consistency
    
    # 特征重要性分数
    feature_importance = torch.mean(model_outputs['feature_weights'], dim=0)
    metrics['feature_importance'] = feature_importance.cpu().numpy()
    
    # 注意力分布熵
    attention_entropy = -(
        model_outputs['attention_weights'] *
        torch.log(model_outputs['attention_weights'] + 1e-10)
    ).sum(1).mean().item()
    metrics['attention_entropy'] = attention_entropy
    
    return metrics

def compute_ensemble_metrics(predictions_list, targets):
    """计算集成模型的指标
    
    Args:
        predictions_list: 多个模型的预测结果列表
        targets: 真实标签
        
    Returns:
        集成指标的字典
    """
    # 将预测转换为numpy数组
    predictions_array = np.array([p.cpu().numpy() for p in predictions_list])
    
    # 计算预测的平均值和标准差
    mean_predictions = np.mean(predictions_array, axis=0)
    std_predictions = np.std(predictions_array, axis=0)
    
    # 计算集成预测
    ensemble_predictions = np.argmax(mean_predictions, axis=1)
    
    # 计算基础指标
    metrics = compute_metrics(ensemble_predictions, targets.cpu().numpy())
    
    # 添加预测不确定性指标
    metrics['prediction_uncertainty'] = std_predictions.mean()
    
    # 计算模型间的一致性
    model_agreements = []
    for i in range(len(predictions_list)):
        for j in range(i + 1, len(predictions_list)):
            agreement = accuracy_score(
                np.argmax(predictions_array[i], axis=1),
                np.argmax(predictions_array[j], axis=1)
            )
            model_agreements.append(agreement)
    
    metrics['model_agreement'] = np.mean(model_agreements)
    
    return metrics 