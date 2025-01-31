import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import logging
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.saman_model import SAMANModel
from configs.config import SAMANConfig
from utils.data_loader import FPLDataset
from utils.metrics import compute_metrics

def setup_logging(config):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.save_path, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_epoch(model, train_loader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    processed_samples = 0
    dataset_size = len(train_loader.dataset)
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        # 修改设备转移逻辑
        processed_features = {}
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                processed_features[k] = v.to(device)
            elif isinstance(v, dict):
                processed_features[k] = {inner_k: inner_v.to(device) if isinstance(inner_v, torch.Tensor) else inner_v 
                                      for inner_k, inner_v in v.items()}
            else:
                processed_features[k] = v
        
        targets = targets.to(device)
        batch_size = targets.size(0)
        processed_samples += batch_size
        
        optimizer.zero_grad()
        output = model(processed_features)
        loss = F.mse_loss(output, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{processed_samples}/{dataset_size} '
                  f'({100. * processed_samples / dataset_size:.0f}%)]\tLoss: {loss.item():.6f}')
    
    return total_loss / dataset_size

def validate(model, val_loader, device):
    """验证模型性能"""
    model.eval()
    val_loss = 0
    predictions = []
    actuals = []
    raw_predictions = []
    raw_actuals = []
    
    with torch.no_grad():
        for features, targets in val_loader:
            # 将特征和目标移动到设备
            processed_features = {
                'historical': {
                    k: v.to(device) for k, v in features['historical'].items()
                },
                'context': {
                    k: v.to(device) for k, v in features['context'].items()
                }
            }
            targets = targets.to(device)
            
            # 前向传播
            output = model(processed_features)
            
            # 计算损失
            loss = F.mse_loss(output, targets)
            val_loss += loss.item()
            
            # 收集标准化的预测和实际值
            predictions.extend(output.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
            
            # 收集原始分数（反标准化）
            raw_pred = val_loader.dataset.inverse_transform_target(output.cpu().numpy())
            raw_true = val_loader.dataset.inverse_transform_target(targets.cpu().numpy())
            raw_predictions.extend(raw_pred)
            raw_actuals.extend(raw_true)
    
    # 计算标准化尺度的评估指标
    val_loss /= len(val_loader)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    # 计算原始分数尺度的评估指标
    raw_mae = mean_absolute_error(raw_actuals, raw_predictions)
    raw_rmse = np.sqrt(mean_squared_error(raw_actuals, raw_predictions))
    raw_r2 = r2_score(raw_actuals, raw_predictions)
    
    return {
        'val_loss': val_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'raw_mae': raw_mae,
        'raw_rmse': raw_rmse,
        'raw_r2': raw_r2
    }

def train(config):
    """训练主函数"""
    logger = setup_logging(config)
    device = torch.device(config.training.device)
    
    # 加载数据
    train_dataset = FPLDataset(config, split='train')
    val_dataset = FPLDataset(config, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    # 初始化模型
    model = SAMANModel(config).to(device)
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.training.lr_scheduler_factor,
        patience=config.training.lr_scheduler_patience,
        verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # 验证
        val_metrics = validate(model, val_loader, device)
        
        # 打印日志
        logger.info(
            f'Epoch {epoch}: '
            f'Train Loss = {train_loss:.4f}, '
            f'Val Loss = {val_metrics["val_loss"]:.4f}\n'
            f'标准化指标 - MAE: {val_metrics["mae"]:.4f}, '
            f'RMSE: {val_metrics["rmse"]:.4f}, '
            f'R2: {val_metrics["r2"]:.4f}\n'
            f'原始分数指标 - MAE: {val_metrics["raw_mae"]:.2f}分, '
            f'RMSE: {val_metrics["raw_rmse"]:.2f}分, '
            f'R2: {val_metrics["raw_r2"]:.4f}'
        )
        
        # 学习率调度
        scheduler.step(val_metrics['val_loss'])
        
        # 保存最佳模型
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(config.save_path, 'best_model.pth'))
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= config.training.early_stopping_patience:
            logger.info(f'Early stopping triggered after {epoch} epochs')
            break
    
    logger.info('Training completed!')

if __name__ == '__main__':
    config = SAMANConfig()
    train(config) 