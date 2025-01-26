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