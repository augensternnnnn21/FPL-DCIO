import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

from models.saman_model import SAMANModel
from configs.config import SAMANConfig
from utils.data_loader import FPLDataset

def setup_logging(config):
    """设置日志"""
    log_file = os.path.join('saman_project', 'checkpoints', 'benchmark.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def prepare_data(config):
    """准备训练和测试数据"""
    # 加载数据集
    train_dataset = FPLDataset(config, split='train')
    val_dataset = FPLDataset(config, split='val')
    test_dataset = FPLDataset(config, split='test')
    
    # 使用DataLoader批量提取特征
    def extract_features_batch(dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=256,  # 使用较大的批量大小
            shuffle=False,
            num_workers=config.training.num_workers
        )
        features = []
        targets = []
        
        print(f"开始提取{len(dataset)}个样本的特征...")
        for batch_idx, (feature_dict, target) in enumerate(dataloader):
            # 展平特征字典
            batch_features = []
            for feature_type in ['historical', 'context']:
                for tensor in feature_dict[feature_type].values():
                    batch_features.append(tensor.numpy().reshape(tensor.shape[0], -1))
            
            # 连接所有特征
            batch_features = np.concatenate(batch_features, axis=1)
            features.append(batch_features)
            targets.append(target.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {(batch_idx + 1) * dataloader.batch_size} 个样本")
        
        return np.concatenate(features), np.concatenate(targets)
    
    print("提取训练集特征...")
    X_train, y_train = extract_features_batch(train_dataset)
    print("提取验证集特征...")
    X_val, y_val = extract_features_batch(val_dataset)
    print("提取测试集特征...")
    X_test, y_test = extract_features_batch(test_dataset)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train_dataset, test_dataset

def evaluate_predictions(y_true, y_pred, scaler=None):
    """评估预测结果"""
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def train_xgboost(X_train, y_train, X_val, y_val):
    """训练XGBoost模型"""
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        callbacks=[xgb.callback.EarlyStopping(rounds=50)]
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    """训练LightGBM模型"""
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    return model

def train_random_forest(X_train, y_train):
    """训练随机森林模型"""
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    )
    model.fit(X_train, y_train)
    return model

def evaluate_saman_model(config, train_dataset, test_dataset):
    """评估SAMAN模型"""
    device = torch.device(config.training.device)
    
    # 加载模型
    model_path = os.path.join('saman_project', 'checkpoints', 'final_model_epoch32.pth')
    model = SAMANModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    # 进行预测
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for features, targets in test_loader:
            processed_features = {
                'historical': {
                    k: v.to(device) for k, v in features['historical'].items()
                },
                'context': {
                    k: v.to(device) for k, v in features['context'].items()
                }
            }
            output = model(processed_features)
            predictions.extend(output.cpu().numpy())
            actuals.extend(targets.numpy())
    
    return np.array(predictions), np.array(actuals)

def historical_baseline(train_data, test_data):
    """历史平均值基线模型"""
    mean_score = np.mean(train_data)
    predictions = np.full(len(test_data), mean_score)
    return predictions

def save_benchmark_results(results, config):
    """保存benchmark结果"""
    output_file = os.path.join('saman_project', 'checkpoints', 'benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    # 1. 初始化
    config = SAMANConfig()
    logger = setup_logging(config)
    logger.info("开始benchmark实验...")
    
    # 2. 准备数据
    logger.info("准备数据...")
    X_train, y_train, X_val, y_val, X_test, y_test, train_dataset, test_dataset = prepare_data(config)
    
    # 3. 训练和评估各个模型
    results = {}
    
    # 3.1 SAMAN模型
    logger.info("评估SAMAN模型...")
    saman_pred, saman_true = evaluate_saman_model(config, train_dataset, test_dataset)
    results['SAMAN'] = evaluate_predictions(saman_true, saman_pred, test_dataset.target_scaler)
    
    # 3.2 XGBoost
    logger.info("训练和评估XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_pred = xgb_model.predict(X_test)
    results['XGBoost'] = evaluate_predictions(y_test, xgb_pred, test_dataset.target_scaler)
    
    # 3.3 LightGBM
    logger.info("训练和评估LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    lgb_pred = lgb_model.predict(X_test)
    results['LightGBM'] = evaluate_predictions(y_test, lgb_pred, test_dataset.target_scaler)
    
    # 3.4 Random Forest
    logger.info("训练和评估Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results['RandomForest'] = evaluate_predictions(y_test, rf_pred, test_dataset.target_scaler)
    
    # 3.5 历史平均值基线
    logger.info("评估历史平均值基线...")
    baseline_pred = historical_baseline(y_train, y_test)
    results['Baseline'] = evaluate_predictions(y_test, baseline_pred, test_dataset.target_scaler)
    
    # 4. 保存和输出结果
    save_benchmark_results(results, config)
    
    # 5. 打印比较结果
    logger.info("\nBenchmark结果比较:")
    logger.info("-" * 60)
    logger.info(f"{'模型':<15} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    logger.info("-" * 60)
    
    for model_name, metrics in results.items():
        logger.info(f"{model_name:<15} {metrics['mae']:>10.4f} {metrics['rmse']:>10.4f} {metrics['r2']:>10.4f}")
    
    logger.info("-" * 60)
    logger.info("Benchmark实验完成！详细结果已保存到benchmark_results.json")

if __name__ == '__main__':
    main() 