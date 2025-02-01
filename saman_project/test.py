import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import seaborn as sns

from models.saman_model import SAMANModel
from configs.config import SAMANConfig
from utils.data_loader import FPLDataset

def setup_logging(config):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.save_path, 'test.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(config, model_path):
    """加载训练好的模型"""
    device = torch.device(config.training.device)
    model = SAMANModel(config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, device

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    predictions = []
    actuals = []
    raw_predictions = []
    raw_actuals = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            # 处理特征
            processed_features = {
                'historical': {
                    k: v.to(device) for k, v in features['historical'].items()
                },
                'context': {
                    k: v.to(device) for k, v in features['context'].items()
                }
            }
            targets = targets.to(device)
            
            # 预测
            output = model(processed_features)
            
            # 收集标准化的预测和实际值
            predictions.extend(output.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
            
            # 收集原始分数
            raw_pred = test_loader.dataset.inverse_transform_target(output.cpu().numpy())
            raw_true = test_loader.dataset.inverse_transform_target(targets.cpu().numpy())
            raw_predictions.extend(raw_pred)
            raw_actuals.extend(raw_true)
    
    return {
        'predictions': np.array(predictions),
        'actuals': np.array(actuals),
        'raw_predictions': np.array(raw_predictions),
        'raw_actuals': np.array(raw_actuals)
    }

def calculate_metrics(results):
    """计算评估指标"""
    # 标准化指标
    mae = mean_absolute_error(results['actuals'], results['predictions'])
    rmse = np.sqrt(mean_squared_error(results['actuals'], results['predictions']))
    r2 = r2_score(results['actuals'], results['predictions'])
    
    # 原始分数指标
    raw_mae = mean_absolute_error(results['raw_actuals'], results['raw_predictions'])
    raw_rmse = np.sqrt(mean_squared_error(results['raw_actuals'], results['raw_predictions']))
    raw_r2 = r2_score(results['raw_actuals'], results['raw_predictions'])
    
    return {
        'standardized': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'raw': {'mae': raw_mae, 'rmse': raw_rmse, 'r2': raw_r2}
    }

def plot_predictions(results, save_path):
    """绘制预测结果可视化"""
    # 1. 预测值vs实际值散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(results['raw_actuals'], results['raw_predictions'], alpha=0.5)
    plt.plot([0, 20], [0, 20], 'r--')  # 理想预测线
    plt.xlabel('实际得分')
    plt.ylabel('预测得分')
    plt.title('预测值 vs 实际值')
    plt.savefig(os.path.join(save_path, 'predictions_scatter.png'))
    plt.close()
    
    # 2. 误差分布直方图
    errors = results['raw_predictions'] - results['raw_actuals']
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50)
    plt.xlabel('预测误差')
    plt.ylabel('频次')
    plt.title('预测误差分布')
    plt.savefig(os.path.join(save_path, 'error_distribution.png'))
    plt.close()
    
    # 3. 预测值和实际值的分布对比
    plt.figure(figsize=(10, 6))
    sns.kdeplot(results['raw_actuals'], label='实际值', color='blue')
    sns.kdeplot(results['raw_predictions'], label='预测值', color='red')
    plt.xlabel('得分')
    plt.ylabel('密度')
    plt.title('预测值和实际值的分布对比')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'distribution_comparison.png'))
    plt.close()

def main():
    # 1. 加载配置
    config = SAMANConfig()
    logger = setup_logging(config)
    
    # 2. 加载模型
    model_path = os.path.join(config.save_path, 'final_model_epoch32.pth')
    model, device = load_model(config, model_path)
    
    # 3. 准备测试数据
    test_dataset = FPLDataset(config, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    # 4. 评估模型
    logger.info("开始模型评估...")
    results = evaluate_model(model, test_loader, device)
    metrics = calculate_metrics(results)
    
    # 5. 记录结果
    logger.info("\n测试集评估结果:")
    logger.info("标准化指标:")
    logger.info(f"- MAE: {metrics['standardized']['mae']:.4f}")
    logger.info(f"- RMSE: {metrics['standardized']['rmse']:.4f}")
    logger.info(f"- R2: {metrics['standardized']['r2']:.4f}")
    
    logger.info("\n原始分数指标:")
    logger.info(f"- MAE: {metrics['raw']['mae']:.2f}分")
    logger.info(f"- RMSE: {metrics['raw']['rmse']:.2f}分")
    logger.info(f"- R2: {metrics['raw']['r2']:.4f}")
    
    # 6. 绘制可视化
    logger.info("\n生成可视化图表...")
    plot_predictions(results, config.save_path)
    
    # 7. 详细分析
    errors = np.abs(results['raw_predictions'] - results['raw_actuals'])
    logger.info("\n预测误差分析:")
    logger.info(f"- 90%的预测误差小于: {np.percentile(errors, 90):.2f}分")
    logger.info(f"- 95%的预测误差小于: {np.percentile(errors, 95):.2f}分")
    logger.info(f"- 最大预测误差: {np.max(errors):.2f}分")
    
    # 8. 保存详细结果
    pd.DataFrame({
        '实际得分': results['raw_actuals'],
        '预测得分': results['raw_predictions'],
        '预测误差': errors
    }).to_csv(os.path.join(config.save_path, 'test_predictions.csv'), index=False)
    
    logger.info("\n测试完成！详细结果已保存到test_predictions.csv")

if __name__ == '__main__':
    main() 