import sys
from pathlib import Path
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from dsn_fpl.src.models.dsn_model import DSNModel
from dsn_fpl.src.core.data.manager import DataManager
from dsn_fpl.src.evaluation.model_tester import ModelTester
from dsn_fpl.src.evaluation.metrics import calculate_metrics

async def run_benchmark(season: str = "2023-24", test_size: float = 0.2):
    """运行DSN模型的benchmark测试
    
    Args:
        season: 测试赛季
        test_size: 测试集比例
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 初始化数据管理器
    data_manager = DataManager(Path(project_root) / "data")
    
    # 初始化模型
    model = DSNModel(data_manager=data_manager).to(device)
    
    # 初始化测试器
    data_path = Path(project_root) / "data"
    tester = ModelTester(data_path, model)
    
    # 加载数据
    season_data = tester.load_season_data(season)
    
    # 准备测试数据
    players_raw = season_data['players_raw']
    
    # 确保所有必要的列都存在
    if 'id' not in players_raw.columns and 'element' in players_raw.columns:
        players_raw['id'] = players_raw['element']
        
    if 'name' not in players_raw.columns:
        if 'first_name' in players_raw.columns and 'second_name' in players_raw.columns:
            players_raw['name'] = players_raw['first_name'] + ' ' + players_raw['second_name']
        elif 'web_name' in players_raw.columns:
            players_raw['name'] = players_raw['web_name']
            
    if 'position' not in players_raw.columns and 'element_type' in players_raw.columns:
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        players_raw['position'] = players_raw['element_type'].map(position_map)
        
    # 创建测试数据字典
    test_data = {
        'player_ids': players_raw['id'].tolist(),
        'player_names': dict(zip(
            players_raw['id'],
            players_raw['name']
        )),
        'positions': dict(zip(
            players_raw['id'],
            players_raw['position']
        )),
        'stats': {
            int(row['id']): {
                'form': float(row.get('form', 0)),
                'fitness': 1.0,  # 默认值
                'momentum': float(row.get('momentum', 0)),
                'potential': float(row.get('potential', 0.5)),
                'minutes': float(row.get('minutes', 0)),
                'total_points': float(row.get('total_points', 0)),
                'goals_scored': float(row.get('goals_scored', 0)),
                'assists': float(row.get('assists', 0)),
                'clean_sheets': float(row.get('clean_sheets', 0)),
                'goals_conceded': float(row.get('goals_conceded', 0)),
                'own_goals': float(row.get('own_goals', 0)),
                'penalties_saved': float(row.get('penalties_saved', 0)),
                'penalties_missed': float(row.get('penalties_missed', 0)),
                'yellow_cards': float(row.get('yellow_cards', 0)),
                'red_cards': float(row.get('red_cards', 0)),
                'saves': float(row.get('saves', 0)),
                'bonus': float(row.get('bonus', 0)),
                'bps': float(row.get('bps', 0)),
                'influence': float(row.get('influence', 0)),
                'creativity': float(row.get('creativity', 0)),
                'threat': float(row.get('threat', 0)),
                'ict_index': float(row.get('ict_index', 0)),
                'selected_by_percent': float(row.get('selected_by_percent', 0))
            }
            for _, row in players_raw.iterrows()
        },
        'env_data': {},  # 环境数据，可以根据需要添加
        'weather_data': {},  # 天气数据，可以根据需要添加
        'media_data': {},  # 媒体数据，可以根据需要添加
        'venue_data': {},  # 场地数据，可以根据需要添加
        'formation': '4-4-2',  # 默认阵型
        'captain_id': players_raw['id'].iloc[0]  # 默认队长
    }
    
    # 评估模型
    logging.info("评估模型性能...")
    model.eval()
    with torch.no_grad():
        predictions = await model.forward(test_data)
        predictions = predictions.cpu().numpy().flatten()  # 展平预测结果
        
    # 获取实际得分并确保维度匹配
    actual_points = players_raw['total_points'].values
    
    # 打印维度信息以便调试
    logging.info(f"预测值维度: {predictions.shape}")
    logging.info(f"真实值维度: {actual_points.shape}")
    
    # 计算评估指标
    metrics = calculate_metrics(actual_points, predictions)
    
    # 运行基准测试
    logging.info("运行基准测试...")
    benchmark_results = {}
    
    # 准备训练数据
    train_data = players_raw.copy()  # 使用当前数据作为训练数据
    
    for name, predictor in tester.benchmarks.items():
        benchmark_pred = predictor(train_data, test_data)
        benchmark_results[name] = calculate_metrics(actual_points, benchmark_pred)
        
    # 输出结果
    logging.info("\nDSN模型性能:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
        
    logging.info("\n基准模型性能:")
    for name, results in benchmark_results.items():
        logging.info(f"\n{name}模型:")
        for metric, value in results.items():
            logging.info(f"{metric}: {value:.4f}")
            
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark_results.log')
        ]
    )
    
    # 运行benchmark测试
    import asyncio
    asyncio.run(run_benchmark()) 