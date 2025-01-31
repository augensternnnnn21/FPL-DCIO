import pandas as pd
import numpy as np
from pathlib import Path
import json
import os

def process_historical_data(data_path, seasons=None):
    """处理历史数据"""
    if seasons is None:
        seasons = ['2023-24', '2022-23', '2021-22']
    
    all_data = []
    for season in seasons:
        season_path = Path(data_path) / season
        if not season_path.exists():
            print(f"跳过不存在的赛季: {season}")
            continue
            
        # 加载球队数据
        teams = pd.read_csv(season_path / 'teams.csv')
        
        # 处理每个gameweek的数据
        for gw in range(1, 39):
            gw_path = season_path / f'gws/gw{gw}.csv'
            if not gw_path.exists():
                continue
                
            # 加载gameweek数据
            gw_df = pd.read_csv(gw_path)
            
            # 转换was_home列
            gw_df['was_home'] = gw_df['was_home'].map({'True': True, 'False': False})
            
            # 转换数值型列
            numeric_columns = [
                'total_points', 'minutes', 'goals_scored', 'assists',
                'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved',
                'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'bonus',
                'bps', 'influence', 'creativity', 'threat', 'ict_index', 'xP',
                'expected_goals', 'expected_assists', 'expected_goal_involvements'
            ]
            
            for col in numeric_columns:
                if col in gw_df.columns:
                    gw_df[col] = pd.to_numeric(gw_df[col], errors='coerce').fillna(0)
            
            # 添加赛季和轮次信息
            gw_df['season'] = season
            gw_df['gameweek'] = gw
            
            all_data.append(gw_df)
    
    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 添加历史表现统计
    combined_data = add_historical_stats(combined_data)
    
    # 添加对手信息
    combined_data = add_opponent_info(combined_data)
    
    return combined_data

def add_historical_stats(df):
    """添加历史统计特征"""
    # 按球员和赛季分组计算滚动统计量
    grouped = df.groupby(['name', 'season'])
    
    # 计算过去5轮的统计量
    for col in ['total_points', 'minutes', 'goals_scored', 'assists', 
                'influence', 'creativity', 'threat']:
        if col in df.columns:
            ma_col = f'{col}_ma5'
            df[ma_col] = grouped[col].transform(
                lambda x: x.astype(float).rolling(5, min_periods=1).mean()
            )
    
    # 计算赛季累计统计量
    for col in ['total_points', 'minutes', 'goals_scored', 'assists']:
        if col in df.columns:
            cum_col = f'{col}_cum'
            df[cum_col] = grouped[col].transform(
                lambda x: x.astype(float).cumsum()
            )
    
    # 计算出场次数
    df['games_played'] = grouped['minutes'].transform(
        lambda x: (x.astype(float) > 0).cumsum()
    )
    
    # 计算形式
    df['form'] = df['total_points_ma5']
    
    return df

def add_opponent_info(df):
    """添加对手信息"""
    # 计算每个球队的强度指标
    team_strength = df.groupby('team').agg({
        'total_points': 'mean',
        'goals_scored': 'mean',
        'goals_conceded': 'mean',
        'clean_sheets': 'mean'
    }).reset_index()
    
    # 计算主场和客场的强度
    home_strength = df[df['was_home'] == True].groupby('team').agg({
        'total_points': 'mean',
        'goals_scored': 'mean',
        'goals_conceded': 'mean',
        'clean_sheets': 'mean'
    }).add_prefix('home_').reset_index()
    
    away_strength = df[df['was_home'] == False].groupby('team').agg({
        'total_points': 'mean',
        'goals_scored': 'mean',
        'goals_conceded': 'mean',
        'clean_sheets': 'mean'
    }).add_prefix('away_').reset_index()
    
    # 合并球队强度信息
    team_strength = pd.merge(team_strength, home_strength, on='team', how='left')
    team_strength = pd.merge(team_strength, away_strength, on='team', how='left')
    
    # 填充数值列的缺失值
    numeric_columns = team_strength.select_dtypes(include=[np.number]).columns
    team_strength[numeric_columns] = team_strength[numeric_columns].fillna(
        team_strength[numeric_columns].mean()
    )
    
    # 计算综合强度指标
    team_strength['strength_overall'] = (
        team_strength['total_points'] * 0.4 +
        team_strength['goals_scored'] * 0.3 -
        team_strength['goals_conceded'] * 0.3
    )
    
    team_strength['strength_overall_home'] = (
        team_strength['home_total_points'] * 0.4 +
        team_strength['home_goals_scored'] * 0.3 -
        team_strength['home_goals_conceded'] * 0.3
    )
    
    team_strength['strength_overall_away'] = (
        team_strength['away_total_points'] * 0.4 +
        team_strength['away_goals_scored'] * 0.3 -
        team_strength['away_goals_conceded'] * 0.3
    )
    
    # 添加对手强度信息
    df = pd.merge(
        df,
        team_strength[['team', 'strength_overall', 'strength_overall_home', 'strength_overall_away']],
        on='team',
        how='left'
    )
    
    df = pd.merge(
        df,
        team_strength[['team', 'strength_overall', 'strength_overall_home', 'strength_overall_away']],
        left_on='opponent_team',
        right_on='team',
        how='left',
        suffixes=('', '_opponent')
    )
    
    # 计算对手强度
    df['opponent_strength'] = np.where(
        df['was_home'],
        df['strength_overall_away_opponent'],
        df['strength_overall_home_opponent']
    ).astype(float)
    
    return df

def prepare_training_data(processed_data, train_seasons=None, val_season=None):
    """准备训练数据"""
    if train_seasons is None:
        train_seasons = ['2021-22', '2022-23']
    if val_season is None:
        val_season = '2023-24'
    
    # 分割训练集和验证集
    train_data = processed_data[processed_data['season'].isin(train_seasons)].copy()
    val_data = processed_data[processed_data['season'] == val_season].copy()
    
    # 创建目录
    os.makedirs('data/processed', exist_ok=True)
    
    # 保存处理后的数据
    train_data.to_csv('data/processed/train.csv', index=False)
    val_data.to_csv('data/processed/val.csv', index=False)
    
    # 保存特征信息
    feature_info = {
        'numerical_features': [
            'total_points', 'minutes', 'goals_scored', 'assists',
            'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved',
            'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'bonus',
            'bps', 'influence', 'creativity', 'threat', 'ict_index', 'xP',
            'total_points_ma5', 'minutes_ma5', 'goals_scored_ma5', 'assists_ma5',
            'influence_ma5', 'creativity_ma5', 'threat_ma5',
            'total_points_cum', 'minutes_cum', 'goals_scored_cum', 'assists_cum',
            'games_played', 'form', 'opponent_strength',
            'strength_overall', 'strength_overall_home', 'strength_overall_away',
            'expected_goals', 'expected_assists', 'expected_goal_involvements'
        ],
        'categorical_features': [
            'position', 'team', 'opponent_team', 'was_home'
        ]
    }
    
    with open('data/processed/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # 打印数据统计信息
    print("\n数据统计信息:")
    print(f"训练集中的球员数量: {train_data['name'].nunique()}")
    print(f"验证集中的球员数量: {val_data['name'].nunique()}")
    print("\n位置分布:")
    print(processed_data['position'].value_counts())
    
    # 打印特征统计
    print("\n数值特征统计:")
    for feature in feature_info['numerical_features']:
        if feature in processed_data.columns:
            stats = processed_data[feature].describe()
            print(f"\n{feature}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
    
    print(f"\n训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    
    print("\n特征列表:")
    print("\nnumerical_features:")
    print(feature_info['numerical_features'])
    print("\ncategorical_features:")
    print(feature_info['categorical_features'])
    
    return train_data, val_data

if __name__ == '__main__':
    # 处理数据
    processed_data = process_historical_data('data')
    train_data, val_data = prepare_training_data(processed_data)
    
    print(f"\n训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print("\n特征列表:")
    for feature_type, features in json.load(open('saman_project/data/processed/feature_info.json')).items():
        print(f"\n{feature_type}:")
        print(features) 