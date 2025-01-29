from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn

class ModelTester:
    """模型测试器"""
    def __init__(self, data_path: Path, model=None):
        self.data_path = data_path
        self.model = model  # 你的模型
        self.results = {}
        self.benchmarks = {
            'average': self.average_predictor,
            'last_season': self.last_season_predictor,
            'position_average': self.position_average_predictor
        }
        
    def load_season_data(self, season: str) -> Dict:
        """加载单个赛季数据"""
        season_path = self.data_path / season
        merged_data_path = self.data_path / 'cleaned_merged_seasons.csv'
        
        print(f"正在加载赛季数据: {season}")
        print(f"数据路径: {season_path}")
        
        # 1. 首先加载原始赛季数据
        try:
            data = {
                'players_raw': pd.read_csv(season_path / 'players_raw.csv', encoding='utf-8'),
                'cleaned_players': pd.read_csv(season_path / 'cleaned_players.csv', encoding='utf-8'),
                'gws': []
            }
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试其他编码
            print("UTF-8编码失败，尝试使用latin-1编码...")
            data = {
                'players_raw': pd.read_csv(season_path / 'players_raw.csv', encoding='latin-1'),
                'cleaned_players': pd.read_csv(season_path / 'cleaned_players.csv', encoding='latin-1'),
                'gws': []
            }
        
        # 2. 处理球员名字
        def create_name_column(df):
            """创建统一的name列"""
            if 'name' not in df.columns:
                if 'first_name' in df.columns and 'second_name' in df.columns:
                    df['name'] = df['first_name'] + ' ' + df['second_name']
                elif 'web_name' in df.columns:
                    df['name'] = df['web_name']
            return df
        
        # 为所有数据框添加name列
        data['cleaned_players'] = create_name_column(data['cleaned_players'])
        data['players_raw'] = create_name_column(data['players_raw'])
        
        # 3. 加载和处理GW数据
        gw_path = season_path / 'gws'
        if gw_path.exists():
            # 加载所有GW数据
            gw_files = sorted(gw_path.glob('gw*.csv'))
            if not gw_files:  # 如果没有单独的GW文件，尝试加载merged_gw
                merged_gw = gw_path / 'merged_gw.csv'
                if merged_gw.exists():
                    print("使用merged_gw.csv")
                    gw_data = self.safe_read_csv(merged_gw)
                    data['gws'] = [create_name_column(gw_data)]
            else:
                print(f"加载 {len(gw_files)} 个GW文件")
                data['gws'] = [create_name_column(self.safe_read_csv(f)) for f in gw_files]
        
        # 4. 从GW数据中提取value信息
        if data['gws']:
            if len(data['gws']) == 1:  # 使用merged_gw
                gw_data = data['gws'][0]
                print("从merged_gw提取value信息")
                value_data = gw_data.groupby('name')['value'].mean().reset_index()
            else:  # 使用单独的GW文件
                print("从单独GW文件提取value信息")
                all_gw_data = pd.concat(data['gws'])
                value_data = all_gw_data.groupby('name')['value'].mean().reset_index()
            
            print(f"找到 {len(value_data)} 个球员的value信息")
            # 将value信息合并到cleaned_players
            data['cleaned_players'] = data['cleaned_players'].merge(
                value_data,
                on='name',
                how='left'
            )
        else:
            print("尝试从players_raw获取value信息")
            # 如果没有GW数据，尝试从players_raw获取value
            if 'now_cost' in data['players_raw'].columns:
                value_data = data['players_raw'][['name', 'now_cost']].copy()
                value_data['value'] = value_data['now_cost'] / 10
                data['cleaned_players'] = data['cleaned_players'].merge(
                    value_data[['name', 'value']],
                    on='name',
                    how='left'
                )
        
        # 打印一些调试信息
        print("\n数据列信息:")
        print("cleaned_players columns:", data['cleaned_players'].columns.tolist())
        if 'name' in data['cleaned_players'].columns:
            print("示例球员名:", data['cleaned_players']['name'].head().tolist())
        
        # 4. 从合并数据中提取球员位置信息
        position_data = pd.read_csv(merged_data_path, usecols=['player_name', 'position'])
        position_data = position_data.groupby('player_name')['position'].agg(
            lambda x: x.mode()[0]
        ).reset_index()
        
        # 5. 处理球员名字匹配
        def clean_name(name):
            return ' '.join(str(name).lower().split())
        
        if 'first_name' in data['cleaned_players'].columns and 'second_name' in data['cleaned_players'].columns:
            data['cleaned_players']['name'] = (
                data['cleaned_players']['first_name'] + ' ' + 
                data['cleaned_players']['second_name']
            )
        
        data['cleaned_players']['clean_name'] = data['cleaned_players']['name'].apply(clean_name)
        position_data['clean_name'] = position_data['player_name'].apply(clean_name)
        
        # 6. 合并位置信息
        data['cleaned_players'] = data['cleaned_players'].merge(
            position_data[['clean_name', 'position']],
            on='clean_name',
            how='left'
        )
        
        # 7. 处理缺失的位置信息
        missing_positions = data['cleaned_players']['position'].isna().sum()
        if missing_positions > 0:
            print(f"警告: {missing_positions} 名球员缺少位置信息")
            if 'element_type' in data['cleaned_players'].columns:
                position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                data['cleaned_players'].loc[
                    data['cleaned_players']['position'].isna(),
                    'position'
                ] = data['cleaned_players'].loc[
                    data['cleaned_players']['position'].isna(),
                    'element_type'
                ].map(position_map)
        
        # 在清理和验证之前，添加球队信息处理
        # 从players_raw中获取球队信息
        if 'team' not in data['cleaned_players'].columns:
            if 'team' in data['players_raw'].columns:
                team_data = data['players_raw'][['name', 'team']].copy()
                data['cleaned_players'] = data['cleaned_players'].merge(
                    team_data,
                    on='name',
                    how='left'
                )
            elif 'team_code' in data['players_raw'].columns:
                # 如果有team_code，需要加载team映射
                team_map_path = self.data_path / 'master_team_list.csv'
                if team_map_path.exists():
                    team_map = pd.read_csv(team_map_path)
                    team_data = data['players_raw'][['name', 'team_code']].copy()
                    team_data = team_data.merge(
                        team_map[['team_code', 'team_name']],
                        on='team_code',
                        how='left'
                    )
                    data['cleaned_players'] = data['cleaned_players'].merge(
                        team_data[['name', 'team_name']].rename(columns={'team_name': 'team'}),
                        on='name',
                        how='left'
                    )
        
        # 添加球队映射
        team_map = {
            1: 'Arsenal',
            2: 'Aston Villa',
            3: 'Bournemouth',
            4: 'Brentford',
            5: 'Brighton',
            6: 'Chelsea',
            7: 'Crystal Palace',
            8: 'Everton',
            9: 'Fulham',
            10: 'Leicester',
            11: 'Leeds',
            12: 'Liverpool',
            13: 'Man City',
            14: 'Man Utd',
            15: 'Newcastle',
            16: 'Nott\'m Forest',
            17: 'Southampton',
            18: 'Spurs',
            19: 'West Ham',
            20: 'Wolves'
        }
        
        if 'team' in data['cleaned_players'].columns:
            data['cleaned_players']['team'] = data['cleaned_players']['team'].map(team_map)
        
        # 8. 清理和验证
        data['cleaned_players'] = data['cleaned_players'].drop(columns=['clean_name'])
        
        required_columns = ['position', 'total_points', 'minutes', 'value']
        missing_columns = [col for col in required_columns if col not in data['cleaned_players'].columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        # 9. 打印数据统计信息
        print("\n数据统计:")
        print(f"球员数量: {len(data['cleaned_players'])}")
        print(f"比赛周数: {len(data['gws'])}")
        print("Value统计:")
        print(data['cleaned_players']['value'].describe())
        
        return data
        
    def evaluate_predictions(self, true_values, predictions, prefix=''):
        """评估预测结果"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # 计算MSE
        mse = mean_squared_error(true_values, predictions)
        
        metrics = {
            f'{prefix}mse': mse,
            f'{prefix}rmse': np.sqrt(mse),  # 手动计算RMSE
            f'{prefix}mae': mean_absolute_error(true_values, predictions),
            f'{prefix}r2': r2_score(true_values, predictions)
        }
        return metrics
    
    def average_predictor(self, train_data, test_data):
        """基准模型1：使用历史平均分"""
        mean_points = train_data['total_points'].mean()
        return np.full(len(test_data), mean_points)
    
    def last_season_predictor(self, train_data, test_data):
        """基准模型2：使用上赛季得分"""
        last_season_points = train_data.groupby('name')['total_points'].last()
        predictions = []
        for player in test_data['name']:
            predictions.append(last_season_points.get(player, last_season_points.mean()))
        return np.array(predictions)
    
    def position_average_predictor(self, train_data, test_data):
        """基准模型3：使用相同位置的平均分"""
        pos_means = train_data.groupby('position')['total_points'].mean()
        return test_data['position'].map(pos_means).fillna(pos_means.mean())
    
    def evaluate_model(self, train_seasons: List[str], test_season: str):
        """评估模型性能"""
        # 加载训练数据
        train_data = pd.concat([
            self.load_season_data(season)['cleaned_players']
            for season in train_seasons
        ])
        
        # 加载测试数据
        test_data_dict = self.load_season_data(test_season)
        test_data = test_data_dict['cleaned_players']
        
        # 实际值
        y_true = test_data['total_points']
        
        # 评估结果
        evaluation = {
            'test_season': test_season,
            'train_seasons': train_seasons,
            'sample_size': len(test_data),
            'metrics': {},
            # 添加数据质量评估
            'data_quality': {
                'player_count': len(test_data),
                'gameweek_count': len(test_data_dict['gws']),
                'missing_values': test_data.isnull().sum().to_dict(),
                'value_ranges': {
                    col: {
                        'min': test_data[col].min(),
                        'max': test_data[col].max(),
                        'mean': test_data[col].mean()
                    }
                    for col in test_data.select_dtypes(include=[np.number]).columns
                }
            },
            # 添加其他分析结果
            'player_analysis': self.analyze_players(test_data_dict),
            'team_analysis': self.analyze_teams(test_data_dict),
            'temporal_analysis': self.analyze_temporal_patterns(test_data_dict)
        }
        
        # 评估基准模型
        for name, benchmark in self.benchmarks.items():
            predictions = benchmark(train_data, test_data)
            metrics = self.evaluate_predictions(y_true, predictions, f'{name}_')
            evaluation['metrics'].update(metrics)
        
        # 评估你的模型
        if self.model is not None:
            # 准备训练数据
            train_features = [
                'minutes', 'goals_scored', 'assists', 'clean_sheets',
                'goals_conceded', 'value', 'position', 'team'
            ]  # 根据你的模型需要的特征调整
            
            # 预处理数据
            X_train = self.model.preprocess_data(train_data[train_features])
            y_train = train_data['total_points']
            X_test = self.model.preprocess_data(test_data[train_features])
            
            # 训练模型
            self.model.train(X_train, y_train)
            
            # 预测
            predictions = self.model.predict(X_test)
            
            # 评估
            metrics = self.evaluate_predictions(y_true, predictions, 'model_')
            evaluation['metrics'].update(metrics)
        
        return evaluation
        
    def run_full_test(self, test_seasons: List[str] = None):
        """运行完整测试"""
        if test_seasons is None:
            test_seasons = ["2022-23"]
        
        for test_season in test_seasons:
            # 使用之前的所有赛季作为训练数据
            available_seasons = [
                "2016-17", "2017-18", "2018-19", "2019-20",
                "2020-21", "2021-22", "2022-23"
            ]
            train_seasons = [s for s in available_seasons if s < test_season]
            
            print(f"\n评估赛季: {test_season}")
            print(f"训练赛季: {train_seasons}")
            
            evaluation = self.evaluate_model(train_seasons, test_season)
            self.results[test_season] = evaluation
            
    def test_data_quality(self, data: Dict) -> Dict:
        """测试数据质量"""
        quality_metrics = {
            'player_count': len(data['cleaned_players']),
            'gameweek_count': len(data['gws']),
            'missing_values': data['cleaned_players'].isnull().sum().to_dict(),
            'value_ranges': {
                col: {
                    'min': data['cleaned_players'][col].min(),
                    'max': data['cleaned_players'][col].max(),
                    'mean': data['cleaned_players'][col].mean()
                }
                for col in data['cleaned_players'].select_dtypes(include=[np.number]).columns
            }
        }
        return quality_metrics
        
    def analyze_players(self, data: Dict) -> Dict:
        """分析球员表现"""
        players = data['cleaned_players']
        
        # 验证position列存在
        if 'position' not in players.columns:
            raise KeyError("找不到position列，请确保数据加载正确")
        
        # 确保total_points和value是数值类型
        players['total_points'] = pd.to_numeric(players['total_points'], errors='coerce')
        players['value'] = pd.to_numeric(players['value'], errors='coerce')
        
        # 打印一些调试信息
        print("\n值类型检查:")
        print("total_points dtype:", players['total_points'].dtype)
        print("value dtype:", players['value'].dtype)
        print("\n空值检查:")
        print("total_points null count:", players['total_points'].isnull().sum())
        print("value null count:", players['value'].isnull().sum())
        
        # 过滤掉无效数据
        valid_players = players.dropna(subset=['total_points', 'value'])
        
        # 计算效率指标
        valid_players['points_per_mil'] = (
            valid_players['total_points'] / valid_players['value']
        ).round(2)
        
        # 修改position_stats的计算方式
        position_stats = valid_players.groupby('position').agg({
            'total_points': ['mean', 'std', 'max'],
            'minutes': 'mean',
            'value': 'mean'
        }).round(2)
        
        # 转换多级索引为单级字典
        position_stats_dict = {}
        for position in position_stats.index:
            position_stats_dict[position] = {
                'total_points': {
                    'mean': position_stats.loc[position, ('total_points', 'mean')],
                    'std': position_stats.loc[position, ('total_points', 'std')],
                    'max': position_stats.loc[position, ('total_points', 'max')]
                },
                'minutes': {
                    'mean': position_stats.loc[position, ('minutes', 'mean')]
                },
                'value': {
                    'mean': position_stats.loc[position, ('value', 'mean')]
                }
            }
        
        analysis = {
            'position_stats': position_stats_dict,
            
            'top_performers': valid_players.nlargest(10, 'total_points')[
                ['name', 'position', 'total_points', 'value']
            ].to_dict('records'),
            
            'value_efficiency': valid_players.nlargest(10, 'points_per_mil')[
                ['name', 'position', 'points_per_mil', 'total_points', 'value']
            ].to_dict('records')
        }
        
        return analysis
        
    def analyze_temporal_patterns(self, data: Dict) -> Dict:
        """分析时间序列模式"""
        gw_stats = []
        for gw_num, gw_data in enumerate(data['gws'], 1):
            gw_stats.append({
                'gw': gw_num,
                'mean_points': gw_data['total_points'].mean(),
                'max_points': gw_data['total_points'].max(),
                'total_minutes': gw_data['minutes'].sum()
            })
            
        return {
            'gameweek_stats': gw_stats,
            'high_scoring_gws': [
                gw['gw'] for gw in gw_stats
                if gw['mean_points'] > np.mean([g['mean_points'] for g in gw_stats]) + 
                   np.std([g['mean_points'] for g in gw_stats])
            ]
        }
        
    def analyze_teams(self, data: Dict) -> Dict:
        """分析球队表现"""
        players = data['cleaned_players']
        
        # 检查team列是否存在
        if 'team' not in players.columns:
            print("警告: 找不到team列，尝试使用替代列...")
            # 尝试其他可能的列名
            team_columns = ['team', 'team_name', 'team_code']
            found_col = None
            for col in team_columns:
                if col in players.columns:
                    players['team'] = players[col]
                    found_col = col
                    break
            
            if not found_col:
                print("错误: 无法找到球队信息")
                return {
                    'team_stats': {},
                    'top_teams': []
                }
        
        # 打印一些调试信息
        print("\n球队信息统计:")
        print("唯一球队数:", players['team'].nunique())
        print("球队列表:", sorted(players['team'].unique()))
        
        team_stats = players.groupby('team').agg({
            'total_points': ['sum', 'mean'],
            'value': 'mean',
            'name': 'count'  # 使用name替代id来计数
        }).round(2)
        
        return {
            'team_stats': team_stats.to_dict(),
            'top_teams': team_stats.nlargest(5, ('total_points', 'sum'))
                        .index.tolist()
        }
        
    def generate_report(self):
        """生成测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("test_reports")
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / f"test_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            for season, results in self.results.items():
                f.write(f"\n{'='*50}\n")
                f.write(f"测试赛季: {season}\n")
                f.write(f"训练赛季: {results['train_seasons']}\n")
                f.write(f"{'='*50}\n\n")
                
                # 模型评估结果
                f.write("1. 模型性能评估\n")
                f.write("\n基准模型性能:\n")
                for metric, value in results['metrics'].items():
                    if metric.startswith(('average_', 'last_season_', 'position_average_')):
                        f.write(f"{metric}: {value:.4f}\n")
                
                f.write("\n你的模型性能:\n")
                for metric, value in results['metrics'].items():
                    if metric.startswith('model_'):
                        f.write(f"{metric}: {value:.4f}\n")
                
                # 数据质量报告
                f.write("\n2. 数据质量指标\n")
                f.write(f"球员数量: {results['data_quality']['player_count']}\n")
                f.write(f"比赛周数: {results['data_quality']['gameweek_count']}\n")
                
                # 球员分析
                f.write("\n3. 球员表现分析\n")
                f.write("\n最佳表现球员 (按总分):\n")
                for i, player in enumerate(results['player_analysis']['top_performers'][:10], 1):
                    f.write(f"{i}. {player['name']} ({player['position']}) - {player['total_points']}分 (价值: {player['value']})\n")
                
                f.write("\n最具性价比球员:\n")
                for i, player in enumerate(results['player_analysis']['value_efficiency'][:10], 1):
                    f.write(f"{i}. {player['name']} ({player['position']}) - {player['points_per_mil']}分/百万 "
                           f"(总分: {player['total_points']}, 价值: {player['value']})\n")
                
                # 位置分析
                f.write("\n各位置统计:\n")
                for pos, stats in results['player_analysis']['position_stats'].items():
                    if isinstance(stats, dict):  # 处理多级索引
                        f.write(f"{pos}:\n")
                        f.write(f"  平均得分: {stats['total_points']['mean']:.1f}\n")
                        f.write(f"  最高得分: {stats['total_points']['max']:.1f}\n")
                        f.write(f"  平均价值: {stats['value']['mean']:.1f}\n")
                
                # 球队分析
                f.write("\n4. 球队分析\n")
                f.write("表现最佳的球队:\n")
                for i, team in enumerate(results['team_analysis']['top_teams'][:5], 1):
                    f.write(f"{i}. {team}\n")
                
                # 时间序列分析
                f.write("\n5. 时间序列分析\n")
                f.write(f"高分轮次: {results['temporal_analysis']['high_scoring_gws']}\n")

    def safe_read_csv(self, file_path):
        """安全地读取CSV文件，处理不同的编码"""
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding='latin-1')

    def analyze_feature_importance(self, model, feature_names):
        """分析特征重要性"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\n特征重要性排名:")
            for f in range(len(feature_names)):
                print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], 
                                     importances[indices[f]]))

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader, val_loader, epochs):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            # 验证阶段
            val_loss = self.evaluate(val_loader)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        return total_loss / len(data_loader)