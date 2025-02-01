import pandas as pd
import numpy as np
from pathlib import Path
import json
import os

def process_historical_data(data_path='data', seasons=None):
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
        teams_path = season_path / 'teams.csv'
        if not teams_path.exists():
            print(f"找不到球队数据文件: {teams_path}")
            continue
            
        teams = pd.read_csv(teams_path)
        
        # 处理每个gameweek的数据
        gw_dir = season_path / 'gws'
        if not gw_dir.exists():
            print(f"找不到比赛周数据目录: {gw_dir}")
            continue
            
        for gw in range(1, 39):
            gw_path = gw_dir / f'gw{gw}.csv'
            if not gw_path.exists():
                continue
                
            print(f"处理 {season} 赛季第 {gw} 轮数据...")
            
            # 加载gameweek数据
            try:
                gw_df = pd.read_csv(gw_path)
            except Exception as e:
                print(f"读取文件 {gw_path} 时出错: {str(e)}")
                continue
            
            # 转换was_home列
            gw_df['was_home'] = gw_df['was_home'].map({'True': True, 'False': False})
            
            # 转换数值型列
            numeric_columns = [
                'total_points', 'minutes', 'goals_scored', 'assists',
                'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved',
                'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'bonus',
                'bps', 'influence', 'creativity', 'threat', 'ict_index',
                'expected_goals', 'expected_assists', 'expected_goal_involvements'
            ]
            
            for col in numeric_columns:
                if col in gw_df.columns:
                    gw_df[col] = pd.to_numeric(gw_df[col], errors='coerce').fillna(0)
            
            # 添加赛季和轮次信息
            gw_df['season'] = season
            gw_df['gameweek'] = gw
            
            all_data.append(gw_df)
    
    if not all_data:
        raise ValueError("没有找到任何可用的数据")
    
    # 合并所有数据并按时间排序
    print("合并所有数据...")
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sort_values(['season', 'gameweek', 'name'])
    
    # 添加历史表现统计
    print("添加历史统计特征...")
    combined_data = add_historical_stats(combined_data)
    
    # 添加对手信息
    print("添加对手信息...")
    combined_data = add_opponent_info(combined_data)
    
    return combined_data

def add_historical_stats(df):
    """添加历史统计特征（只使用历史数据）"""
    # 按球员和赛季分组，并确保按时间排序
    df = df.sort_values(['name', 'season', 'gameweek'])
    grouped = df.groupby(['name', 'season'])
    
    # 计算历史统计量（使用shift确保不使用当前轮次的数据）
    for col in ['total_points', 'minutes', 'goals_scored', 'assists', 
                'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved',
                'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'bonus',
                'bps', 'influence', 'creativity', 'threat', 'ict_index']:
        if col in df.columns:
            # 使用shift(1)确保不包含当前轮次的数据
            shifted_values = grouped[col].shift(1)
            # 计算移动平均
            df[f'{col}_ma5'] = shifted_values.rolling(5, min_periods=1).mean()
            # 计算累计值
            df[f'{col}_cum'] = shifted_values.cumsum()
            # 计算标准差
            df[f'{col}_std5'] = shifted_values.rolling(5, min_periods=1).std()
    
    # 计算出场率（基于历史数据）
    # 创建一个临时列来标记是否出场
    df['played'] = (df['minutes'].shift(1) > 0).astype(int)
    # 按球员和赛季分组计算累计出场次数
    df['games_played'] = df.groupby(['name', 'season'])['played'].cumsum()
    # 删除临时列
    df.drop('played', axis=1, inplace=True)
    
    # 计算形式（使用过去5轮的平均分）
    df['form'] = df['total_points_ma5']
    
    # 添加趋势指标
    df['points_trend'] = df.groupby(['name', 'season'])['total_points_ma5'].diff()
    
    # 填充缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def add_opponent_info(df):
    """添加对手信息（只使用历史数据）"""
    # 按赛季和轮次分组计算球队的历史表现
    df = df.sort_values(['season', 'gameweek', 'team'])
    team_stats = df.groupby(['team', 'season', 'gameweek']).agg({
        'total_points': 'mean',
        'goals_scored': 'mean',
        'goals_conceded': 'mean',
        'clean_sheets': 'mean'
    }).reset_index()
    
    # 计算历史统计量
    team_stats = team_stats.sort_values(['team', 'season', 'gameweek'])
    grouped = team_stats.groupby(['team', 'season'])
    
    for col in ['total_points', 'goals_scored', 'goals_conceded', 'clean_sheets']:
        # 使用shift(1)确保不包含当前轮次的数据
        shifted_values = grouped[col].shift(1)
        # 计算移动平均
        team_stats[f'{col}_ma5'] = shifted_values.rolling(5, min_periods=1).mean()
        # 计算标准差
        team_stats[f'{col}_std5'] = shifted_values.rolling(5, min_periods=1).std()
    
    # 分别计算主场和客场的历史表现
    for home in [True, False]:
        prefix = 'home' if home else 'away'
        mask = df['was_home'] == home
        venue_stats = df[mask].groupby(['team', 'season', 'gameweek']).agg({
            'total_points': 'mean',
            'goals_scored': 'mean',
            'goals_conceded': 'mean',
            'clean_sheets': 'mean'
        }).reset_index()
        
        # 计算历史统计量
        venue_stats = venue_stats.sort_values(['team', 'season', 'gameweek'])
        grouped = venue_stats.groupby(['team', 'season'])
        
        for col in ['total_points', 'goals_scored', 'goals_conceded', 'clean_sheets']:
            shifted_values = grouped[col].shift(1)
            venue_stats[f'{col}_ma5'] = shifted_values.rolling(5, min_periods=1).mean()
            venue_stats[f'{col}_std5'] = shifted_values.rolling(5, min_periods=1).std()
        
        # 重命名列并合并
        venue_stats = venue_stats.add_prefix(f'{prefix}_')
        venue_stats = venue_stats.rename(columns={
            f'{prefix}_team': 'team',
            f'{prefix}_season': 'season',
            f'{prefix}_gameweek': 'gameweek'
        })
        
        team_stats = pd.merge(
            team_stats, 
            venue_stats, 
            on=['team', 'season', 'gameweek'],
            how='left'
        )
    
    # 计算综合强度指标（使用历史数据）
    team_stats['strength_overall'] = (
        team_stats['total_points_ma5'] * 0.4 +
        team_stats['goals_scored_ma5'] * 0.3 -
        team_stats['goals_conceded_ma5'] * 0.3
    )
    
    team_stats['strength_overall_home'] = (
        team_stats['home_total_points_ma5'] * 0.4 +
        team_stats['home_goals_scored_ma5'] * 0.3 -
        team_stats['home_goals_conceded_ma5'] * 0.3
    )
    
    team_stats['strength_overall_away'] = (
        team_stats['away_total_points_ma5'] * 0.4 +
        team_stats['away_goals_scored_ma5'] * 0.3 -
        team_stats['away_goals_conceded_ma5'] * 0.3
    )
    
    # 合并球队强度信息到主数据框
    df = pd.merge(
        df,
        team_stats[['team', 'season', 'gameweek', 'strength_overall', 
                   'strength_overall_home', 'strength_overall_away']],
        on=['team', 'season', 'gameweek'],
        how='left'
    )
    
    # 添加对手强度信息
    df = pd.merge(
        df,
        team_stats[['team', 'season', 'gameweek', 'strength_overall', 
                   'strength_overall_home', 'strength_overall_away']],
        left_on=['opponent_team', 'season', 'gameweek'],
        right_on=['team', 'season', 'gameweek'],
        how='left',
        suffixes=('', '_opponent')
    )
    
    # 计算对手强度（基于历史数据）
    df['opponent_strength'] = np.where(
        df['was_home'],
        df['strength_overall_away_opponent'],
        df['strength_overall_home_opponent']
    ).astype(float)
    
    # 填充缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def prepare_training_data(processed_data, train_seasons=None, val_season=None, test_season=None):
    """准备训练数据"""
    if train_seasons is None:
        train_seasons = ['2018-19', '2019-20', '2020-21', '2021-22']  # 使用更多的训练赛季
    if val_season is None:
        val_season = '2022-23'
    if test_season is None:
        test_season = '2023-24'
    
    # 分割训练集、验证集和测试集
    train_data = processed_data[processed_data['season'].isin(train_seasons)].copy()
    val_data = processed_data[processed_data['season'] == val_season].copy()
    test_data = processed_data[processed_data['season'] == test_season].copy()
    
    # 创建目录
    os.makedirs('saman_project/data/processed', exist_ok=True)
    
    # 保存处理后的数据
    train_data.to_csv('saman_project/data/processed/train.csv', index=False)
    val_data.to_csv('saman_project/data/processed/val.csv', index=False)
    test_data.to_csv('saman_project/data/processed/test.csv', index=False)
    
    # 保存特征信息
    feature_info = {
        'numerical_features': [
            'total_points_ma5', 'minutes_ma5', 'goals_scored_ma5', 'assists_ma5',
            'clean_sheets_ma5', 'goals_conceded_ma5', 'own_goals_ma5',
            'penalties_saved_ma5', 'penalties_missed_ma5', 'yellow_cards_ma5',
            'red_cards_ma5', 'saves_ma5', 'bonus_ma5', 'bps_ma5',
            'influence_ma5', 'creativity_ma5', 'threat_ma5',
            'total_points_cum', 'minutes_cum', 'goals_scored_cum', 'assists_cum',
            'total_points_std5', 'minutes_std5', 'goals_scored_std5',
            'games_played', 'form', 'points_trend',
            'opponent_strength', 'strength_overall', 'strength_overall_home',
            'strength_overall_away'
        ],
        'categorical_features': [
            'position', 'team', 'opponent_team', 'was_home'
        ]
    }
    
    with open('saman_project/data/processed/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # 打印数据统计信息
    print("\n数据统计信息:")
    print(f"训练集中的球员数量: {train_data['name'].nunique()}")
    print(f"验证集中的球员数量: {val_data['name'].nunique()}")
    print(f"测试集中的球员数量: {test_data['name'].nunique()}")
    
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
    print(f"测试集大小: {len(test_data)}")
    
    print("\n特征列表:")
    print("\nnumerical_features:")
    print(feature_info['numerical_features'])
    print("\ncategorical_features:")
    print(feature_info['categorical_features'])
    
    return train_data, val_data, test_data

if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs('saman_project/data/processed', exist_ok=True)
    
    try:
        # 处理历史数据
        print("开始处理历史数据...")
        processed_data = process_historical_data('data')
        
        # 准备训练数据
        print("\n准备训练数据...")
        train_data, val_data, test_data = prepare_training_data(processed_data)
        
        print("\n数据处理完成!")
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")
        print(f"测试集大小: {len(test_data)}")
        
    except Exception as e:
        print(f"\n❌ 数据处理失败: {str(e)}")
        raise 