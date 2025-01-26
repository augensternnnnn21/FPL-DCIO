from config import BASE_DATA_PATH, PROCESSED_DATA_PATH, NUM_PLAYERS

import os
import pandas as pd
import pickle

def load_and_preprocess_season(season):
    """加载和预处理单个赛季的数据"""
    season_path = os.path.join(BASE_DATA_PATH, season, 'gws')
    print(f"Loading data for season {season_path}...")
    
    # 调试信息
    print("\nDEBUG INFO:")
    print(f"BASE_DATA_PATH: {BASE_DATA_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for season data in: {season_path}")
    print(f"Directory exists: {os.path.exists(season_path)}")
    
    if os.path.exists(season_path):
        print("Files in directory:")
        print(os.listdir(season_path))
    
    # 加载所有比赛周数据
    all_gw_data = []
    total_gws = 0
    
    # 尝试不同的编码
    encodings = ['latin1', 'iso-8859-1', 'cp1252', 'utf-8']
    
    # 遍历所有比赛周文件
    for gw in range(1, 39):  # 假设有38个比赛周
        gw_file = os.path.join(season_path, f'gw{gw}.csv')
        if os.path.exists(gw_file):
            print(f"Processing gameweek {gw}")
            # 尝试不同的编码
            for encoding in encodings:
                try:
                    gw_data = pd.read_csv(gw_file, encoding=encoding)
                    gw_data['gameweek'] = gw
                    all_gw_data.append(gw_data)
                    total_gws += 1
                    print(f"Successfully loaded gameweek {gw} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing gameweek {gw}: {e}")
                    break
    
    print(f"Total gameweeks loaded: {total_gws}")
    
    if not all_gw_data:
        raise ValueError(f"No gameweek data could be loaded for season {season_path}")
    
    # 合并所有数据
    season_data = pd.concat(all_gw_data, ignore_index=True)
    
    # 基本数据清理
    season_data = season_data.fillna(0)
    
    # 保存处理后的数据
    output_path = os.path.join(BASE_DATA_PATH, season, f'{season}_processed.csv')
    season_data.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved processed data to {output_path}")
    
    return season_data

if __name__ == "__main__":
    # 预处理所有赛季数据
    seasons = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    for season in seasons:
        print(f"Processing {season}...")
        load_and_preprocess_season(season)

import pandas as pd
import os

def preprocess_data(data_path: str) -> pd.DataFrame:
    """
    预处理FPL数据
    Args:
        data_path: 数据文件路径
    Returns:
        处理后的DataFrame
    """
    print(f"\nLoading data from {data_path}")
    data = pd.read_csv(data_path)
    print(f"Successfully loaded data with {len(data)} rows")
    
    # 1. 处理比赛周(gw)信息
    print("Processing gameweek information...")
    if 'gw' not in data.columns:
        if 'round' in data.columns:
            data['gw'] = data['round']
        elif 'gameweek' in data.columns:
            data['gw'] = data['gameweek']
        else:
            raise ValueError("No gameweek information found in data")
    
    # 2. 处理球队信息
    print("Processing team information...")
    print("Available columns:", data.columns.tolist())  # 调试信息
    
    # 首先检查是否已经有team列
    if 'team' not in data.columns:
        if 'team_x' in data.columns:
            print("Using team_x column...")
            unique_teams = data['team_x'].unique()
            print(f"Unique teams in team_x: {unique_teams}")
            
            # 创建球队名称标准化映射
            team_mapping = {}
            for team in unique_teams:
                if pd.isna(team) or team == 0:
                    continue
                clean_name = str(team).strip().replace(' ', '_').upper()
                team_mapping[team] = clean_name
            
            # 应用映射
            data['team'] = data['team_x'].map(team_mapping)
        else:
            raise ValueError("No team information found (tried: team, team_x)")
    
    # 验证球队信息
    if data['team'].isna().any():
        missing_count = data['team'].isna().sum()
        total_count = len(data)
        print(f"\nWarning: {missing_count} out of {total_count} team values are missing")
        print("\nSample of rows with missing team values:")
        print(data[data['team'].isna()].head())
        raise ValueError("Some team values are still missing after processing")
    
    # 3. 处理位置信息
    print("\nProcessing position information...")
    position_mapping = {
        'GKP': 'GK',
        'GK': 'GK',
        'DEF': 'DEF',
        'MID': 'MID',
        'FWD': 'FWD',
        'FOR': 'FWD'
    }
    data['position'] = data['position'].map(position_mapping)
    
    # 验证位置信息
    if data['position'].isna().any():
        invalid_positions = data['position'].unique()
        raise ValueError(f"Some players have invalid positions. Found values: {invalid_positions}")
    
    # 4. 确保数值列的类型正确
    numeric_columns = ['value', 'total_points', 'gw']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # 5. 处理缺失值
    data = data.fillna(0)
    
    # 6. 打印数据信息
    print("\nProcessed data info:")
    print(f"Total rows: {len(data)}")
    print(f"Columns: {list(data.columns)}")
    print("\nTeam distribution:")
    print(data['team'].value_counts().head())
    print("\nPosition distribution:")
    print(data['position'].value_counts())
    print("\nGameweek distribution:")
    print(data['gw'].value_counts().sort_index().head())
    
    return data