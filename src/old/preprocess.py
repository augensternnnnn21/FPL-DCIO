import os
import pandas as pd
import numpy as np
from pathlib import Path
from config import BASE_DATA_PATH, PROCESSED_DATA_PATH, NUM_PLAYERS

# 修正数据路径
BASE_DATA_PATH = Path(__file__).parent.parent.parent / "data"  # 回到项目根目录的data文件夹
print(f"\nDEBUG INFO:")
print(f"BASE_DATA_PATH: {BASE_DATA_PATH}")
print(f"Current working directory: {os.getcwd()}")

def load_and_preprocess_season(season):
    """加载和预处理单个赛季的数据"""
    print(f"\nProcessing {season}...")
    
    # 构建赛季数据路径
    season_path = BASE_DATA_PATH / season / "gws"
    print(f"Looking for season data in: {season_path}")
    print(f"Directory exists: {season_path.exists()}")
    
    if not season_path.exists():
        season_path = BASE_DATA_PATH  # 如果没有子目录结构，直接使用基础路径
    
    print(f"Loading data for season {season}...")
    print(f"Looking for files in: {season_path}")
    
    # 加载所有比赛周数据
    all_gws = []
    for gw in range(1, 39):  # FPL赛季有38轮
        try:
            print(f"Processing gameweek {gw}")
            # 尝试不同的文件名格式
            possible_files = [
                f"gw{gw}.csv",
                f"{season}_gw{gw}.csv",
                f"gw_{gw}.csv"
            ]
            
            loaded = False
            for filename in possible_files:
                file_path = season_path / filename
                if file_path.exists():
                    gw_data = pd.read_csv(file_path)
                    gw_data['GW'] = gw
                    gw_data['season'] = season
                    all_gws.append(gw_data)
                    loaded = True
                    break
                    
            if not loaded:
                print(f"Warning: Could not find gameweek {gw} data")
                
        except Exception as e:
            print(f"Error loading gameweek {gw}: {str(e)}")
            continue
    
    print(f"Total gameweeks loaded: {len(all_gws)}")
    
    if not all_gws:
        raise ValueError(f"No gameweek data could be loaded for season {season}")
    
    # 合并所有比赛周数据
    season_data = pd.concat(all_gws, ignore_index=True)
    
    # 基本数据清洗
    season_data = clean_data(season_data)
    
    # 保存处理后的数据
    output_path = BASE_DATA_PATH / "processed" / f"{season}_processed.csv"
    output_path.parent.mkdir(exist_ok=True)
    season_data.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    return season_data

def clean_data(df):
    """清洗数据"""
    # 统一列名
    df = df.rename(columns={
        'name': 'player_name',
        'position': 'pos',
        'value': 'price',
        'total_points': 'points'
    })
    
    # 处理缺失值
    df = df.fillna({
        'minutes': 0,
        'goals_scored': 0,
        'assists': 0,
        'clean_sheets': 0,
        'goals_conceded': 0,
        'bonus': 0,
        'bps': 0
    })
    
    # 确保数值类型正确
    numeric_columns = [
        'minutes', 'goals_scored', 'assists', 
        'clean_sheets', 'goals_conceded', 'bonus',
        'bps', 'points', 'price'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

# 确保添加 GW 列
def preprocess_data(df):
    # 如果没有 GW 列，添加它
    if 'GW' not in df.columns:
        # 假设文件名包含 gameweek 信息，如 gw1.csv, gw2.csv
        gw = int(df.name.split('gw')[1].split('.')[0])
        df['GW'] = gw
    
    # 确保列名统一使用大写
    df.columns = [col.upper() for col in df.columns]
    
    return df

# 在预处理后添加数据验证
def validate_data(df):
    required_columns = ['GW']  # 添加其他必需的列
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")
        
    print("数据列:", df.columns.tolist())
    print("数据样本:\n", df.head())

if __name__ == "__main__":
    # 处理多个赛季
    seasons = [
        "2016-17",
        "2017-18",
        "2018-19",
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24"
    ]
    
    all_seasons_data = []
    for season in seasons:
        try:
            season_data = load_and_preprocess_season(season)
            all_seasons_data.append(season_data)
        except Exception as e:
            print(f"Error processing season {season}: {str(e)}")
            continue
    
    if all_seasons_data:
        # 合并所有赛季数据
        all_data = pd.concat(all_seasons_data, ignore_index=True)
        
        # 保存合并后的数据
        output_path = BASE_DATA_PATH / "cleaned_merged_seasons.csv"
        all_data.to_csv(output_path, index=False)
        print(f"\nSaved merged data to {output_path}")
        print(f"Total records: {len(all_data)}")
    else:
        print("\nNo data could be processed!")

    # 在主处理流程中调用验证
    df = preprocess_data(all_data)
    validate_data(df)