import os
import pandas as pd
import pickle
from config import BASE_DATA_PATH, PROCESSED_DATA_PATH, NUM_PLAYERS

def load_and_preprocess_season(season: str):
    """加载并预处理单个赛季数据"""
    print(f"\nDEBUG INFO:")
    print(f"BASE_DATA_PATH: {BASE_DATA_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    
    # 检查数据目录结构
    season_path = os.path.join(BASE_DATA_PATH, season, "gws")
    print(f"Looking for season data in: {season_path}")
    
    # 验证目录结构
    if not os.path.exists(BASE_DATA_PATH):
        raise FileNotFoundError(f"基础数据目录不存在: {BASE_DATA_PATH}")
    
    if not os.path.exists(season_path):
        raise FileNotFoundError(f"赛季数据目录不存在: {season_path}")
        
    # 验证至少有一个比赛周的数据
    gw1_path = os.path.join(season_path, "gw1.csv")
    if not os.path.exists(gw1_path):
        raise FileNotFoundError(f"找不到第一周的数据文件: {gw1_path}")
    
    all_gw = []
    print(f"Loading data for season {season}...")
    
    # 在读取数据之前打印文件路径
    print(f"Looking for files in: {BASE_DATA_PATH}")
    
    # 尝试不同的编码
    encodings = ['latin1', 'iso-8859-1', 'cp1252']
    
    # 在添加到 all_gw 之前打印每个游戏周的数据
    for gw in range(1, 39):
        try:
            # 打印当前正在处理的游戏周
            print(f"Processing gameweek {gw}")
            file_path = os.path.join(BASE_DATA_PATH, f"{season}/gws/gw{gw}.csv")
            if os.path.exists(file_path):
                # 尝试不同的编码直到成功
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        df['gw'] = gw
                        print(f"Gameweek {gw} data shape: {df.shape} (using {encoding} encoding)")
                        all_gw.append(df)
                        break  # 如果成功读取，跳出编码尝试循环
                    except UnicodeDecodeError:
                        continue  # 如果这个编码不行，尝试下一个
                    except Exception as e:
                        print(f"Unexpected error reading gameweek {gw}: {str(e)}")
                        break
        except Exception as e:
            print(f"Error processing gameweek {gw}: {str(e)}")
    
    print(f"Total gameweeks loaded: {len(all_gw)}")
    
    if len(all_gw) == 0:
        raise ValueError(f"No gameweek data could be loaded for season {season}")
    
    # 首先加载球员基本信息
    players_file = os.path.join(BASE_DATA_PATH, season, "players_raw.csv")
    players_df = pd.read_csv(players_file, encoding='utf-8')
    
    # 打印一些调试信息
    print("\nPlayers data info:")
    print("Players file columns:", players_df.columns.tolist())
    print("Sample of players_df:")
    print(players_df[['id', 'element_type', 'web_name']].head())
    
    # 合并所有比赛数据
    season_df = pd.concat(all_gw, ignore_index=True)
    
    print("\nGame data info:")
    print("Game data columns:", season_df.columns.tolist())
    print("Sample of season_df:")
    print(season_df[['name', 'element']].head())
    
    # 删除重复行
    season_df = season_df.drop_duplicates()
    
    # 特征工程
    features = [
        'name', 'element', 'opponent_team', 'was_home', 'fixture',
        'value', 'total_points', 'minutes', 'goals_scored', 
        'assists', 'clean_sheets', 'gw'
    ]
    
    season_df = season_df[features]

    # preprocess.py 中增加新特征
    # 添加过去3个比赛周的得分均值
    season_df['3_gw_avg'] = season_df.groupby('name')['total_points'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    # 添加是否受伤特征
    season_df['is_injured'] = (season_df['minutes'] == 0).astype(float)


    
    # 添加 team 列（基于 was_home 和 opponent_team）
    def get_team(row):
        if row['was_home']:
            return row['fixture'] // 10  # 主队ID
        else:
            return row['opponent_team']
    
    season_df['team'] = season_df.apply(get_team, axis=1)
    
    # 创建element到element_type的映射
    element_type_map = players_df.set_index('id')['element_type'].to_dict()
    
    # 使用element来合并位置信息
    season_df['element_type'] = season_df['element'].map(element_type_map)
    
    # 将 element_type 映射到位置
    position_mapping = {
        1: 'GK',  # 守门员
        2: 'DEF', # 后卫
        3: 'MID', # 中场
        4: 'FWD'  # 前锋
    }
    # 修复位置映射错误（确保element_type正确映射）
    season_df['position'] = season_df['element_type'].map(position_mapping).fillna('UNK')
    season_df = season_df[season_df['position'] != 'UNK']  # 严格过滤未知位置
    
    # 删除没有位置信息的行
    print("\nBefore dropping NA positions:")
    print(f"Total rows: {len(season_df)}")
    print("Position distribution:")
    print(season_df['position'].value_counts(dropna=False))
    
    season_df = season_df.dropna(subset=['position'])
    
    print("\nAfter dropping NA positions:")
    print(f"Total rows: {len(season_df)}")
    print("Position distribution:")
    print(season_df['position'].value_counts())
    
    # 删除不再需要的列
    season_df = season_df.drop(['opponent_team', 'was_home', 'fixture', 'element', 'element_type'], axis=1)
    
    # 标准化
    # 修改价格标准化逻辑（原值单位为百万*10）
    season_df['value'] = season_df['value'] / 10  # 将价格单位转换为百万
    season_df['total_points'] = season_df['total_points']  # 原值保留
    season_df['minutes'] = season_df['minutes'] / 90
    
    # 筛选活跃球员（按总出场次数排序取前NUM_PLAYERS名）
    active_players = season_df.groupby('name')['minutes'].sum().sort_values(ascending=False).head(NUM_PLAYERS).index
    season_df = season_df[season_df['name'].isin(active_players)]
    
    # 在保存之前打印一些统计信息
    print("\nProcessed data summary:")
    print(f"Total rows: {len(season_df)}")
    print("\nPosition distribution:")
    print(season_df['position'].value_counts())
    print("\nTeam distribution:")
    print(season_df['team'].value_counts())
    print(f"\nUnique players: {len(season_df['name'].unique())}")
    print("\nSample of processed data:")
    print(season_df.head())
    
    # 保存处理后的数据
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    with open(os.path.join(PROCESSED_DATA_PATH, f"{season}_processed.pkl"), 'wb') as f:
        pickle.dump(season_df, f)
    
if __name__ == "__main__":
    # 预处理所有赛季数据
    seasons = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    for season in seasons:
        print(f"Processing {season}...")
        load_and_preprocess_season(season)