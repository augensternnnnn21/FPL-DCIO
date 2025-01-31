import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from dsn_fpl.src.models.dsn_model_improved import DSNModelImproved
from dsn_fpl.src.core.data.manager import DataManager
import logging

logging.basicConfig(level=logging.INFO)

def prepare_data(data_manager):
    """准备训练和测试数据"""
    # 获取最新赛季的数据
    latest_season = data_manager.historical_data['players']['season'].max()
    all_data = data_manager.historical_data['players'][
        data_manager.historical_data['players']['season'] == latest_season
    ]
    
    # 选择特征 - 确保与模型输入维度匹配
    features = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
        'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity',
        'threat', 'ict_index', 'selected_by_percent', 'form'
    ]
    
    # 准备特征和目标值
    X = all_data[features].fillna(0)
    y = all_data['total_points'].fillna(0)
    
    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    logging.info(f"\n{model_name} Performance:")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"R2 Score: {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'r2': r2}

def train_and_evaluate_dsn(data_manager, X_test, y_test):
    """训练和评估DSN模型"""
    # 初始化模型
    model = DSNModelImproved(config={'input_dim': X_test.shape[1]})  # 使用实际的特征维度
    
    try:
        # 加载训练好的模型
        model.load_state_dict(torch.load('best_model.pth'))
        logging.info("成功加载训练好的DSN模型")
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        return {'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')}
    
    model.eval()
    
    # 准备测试数据
    test_data = {
        'player_ids': list(range(len(X_test))),
        'features': torch.FloatTensor(X_test),
        'targets': torch.FloatTensor(y_test.values)
    }
    
    # 预测
    with torch.no_grad():
        predictions = model(test_data)
        
    # 转换预测值
    y_pred = predictions.numpy().flatten()
    
    # 评估
    return evaluate_model(y_test, y_pred, "DSN Model (Trained)")

def benchmark():
    """执行benchmark测试"""
    # 初始化数据管理器
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data"
    data_manager = DataManager(data_path)
    
    # 准备数据
    X_train, X_test, y_train, y_test, scaler = prepare_data(data_manager)
    
    # 确保所有模型使用相同的测试数据
    logging.info(f"\n测试集大小: {len(X_test)} 样本")
    
    # 1. Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")
    
    # 2. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_metrics = evaluate_model(y_test, lr_pred, "Linear Regression")
    
    # 3. SVR
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(X_test)
    svr_metrics = evaluate_model(y_test, svr_pred, "SVR")
    
    # 4. XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = evaluate_model(y_test, xgb_pred, "XGBoost")
    
    # 5. LightGBM
    lgb_model = lgb.LGBMRegressor(random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_metrics = evaluate_model(y_test, lgb_pred, "LightGBM")
    
    # 6. DSN Model
    dsn_metrics = train_and_evaluate_dsn(data_manager, X_test, y_test)
    
    # 汇总结果
    results = pd.DataFrame({
        'Random Forest': rf_metrics,
        'Linear Regression': lr_metrics,
        'SVR': svr_metrics,
        'XGBoost': xgb_metrics,
        'LightGBM': lgb_metrics,
        'DSN': dsn_metrics
    }).T
    
    logging.info("\nBenchmark Results:")
    logging.info(results)
    
    # 保存结果
    results.to_csv('benchmark_results.csv')

if __name__ == "__main__":
    benchmark() 