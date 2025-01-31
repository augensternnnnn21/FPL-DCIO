# dsn_fpl/src/data/dataset_generator.py

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
from .collector import FPLDataCollector, EnvironmentalData, NewsSource
from .weather_collector import WeatherCollector
from .media_collector import MediaCollector
from .venue_collector import VenueCollector
from .config import Config
import asyncio

class DSNDatasetGenerator:
    def __init__(self, data_path: Path = None):
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent.parent.parent / "data"
        self.data_path = Path(data_path)
        print(f"Initializing DSNDatasetGenerator with data path: {self.data_path}")
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        # 初始化数据收集器
        self.collector = FPLDataCollector()
        self.weather_collector = WeatherCollector(api_key=Config().weather_api_key)
        self.media_collector = MediaCollector(news_api_key=Config().news_api_key)
        self.venue_collector = VenueCollector()
        self.current_season = "2023-24"
        
    def load_historical_data(self) -> Dict:
        """从本地加载FPL历史数据"""
        seasons = ['2016-17', '2017-18', '2018-19', '2019-20', 
                  '2020-21', '2021-22', '2022-23', '2023-24']
        
        data = {
            'players': [],
            'gw': [],
            'fixtures': {},
            'players_raw': []
        }
        
        def safe_read_csv(file_path: Path) -> pd.DataFrame:
            """安全读取CSV文件"""
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return pd.DataFrame()
                
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                print(f"Successfully read {file_path}")
                return df
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                    print(f"Successfully read {file_path} with latin-1 encoding")
                    return df
                except Exception as e:
                    print(f"Error reading {file_path} with latin-1 encoding: {e}")
                    return pd.DataFrame()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return pd.DataFrame()
        
        successful_seasons = []
        for season in seasons:
            try:
                print(f"\nProcessing season {season}...")
                season_path = self.data_path / season
                
                if not season_path.exists():
                    print(f"Season directory not found: {season_path}")
                    continue
                    
                # 从本地读取数据
                players_raw = safe_read_csv(season_path / 'players_raw.csv')
                cleaned_players = safe_read_csv(season_path / 'cleaned_players.csv')
                
                # 检查gws目录
                gw_path = season_path / 'gws'
                if not gw_path.exists():
                    print(f"GW directory not found: {gw_path}")
                    continue
                    
                merged_gw = safe_read_csv(gw_path / 'merged_gw.csv')
                
                # 验证数据完整性
                if players_raw.empty or cleaned_players.empty or merged_gw.empty:
                    print(f"Skipping season {season} due to incomplete data")
                    continue
                    
                print(f"\nData shapes for season {season}:")
                print(f"players_raw: {players_raw.shape}")
                print(f"cleaned_players: {cleaned_players.shape}")
                print(f"merged_gw: {merged_gw.shape}")
                
                # 处理球员名字
                if 'web_name' in players_raw.columns:
                    cleaned_players['name'] = players_raw['web_name']
                elif 'first_name' in players_raw.columns and 'second_name' in players_raw.columns:
                    cleaned_players['name'] = players_raw['first_name'] + ' ' + players_raw['second_name']
                else:
                    print(f"Warning: Could not process player names for season {season}")
                    print("Available columns in players_raw:", players_raw.columns.tolist())
                    continue
                
                # 处理位置信息
                position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                if 'element_type' in players_raw.columns:
                    cleaned_players['position'] = players_raw['element_type'].map(position_map)
                else:
                    print(f"Warning: No position information for season {season}")
                    print("Available columns in players_raw:", players_raw.columns.tolist())
                    continue
                
                # 添加唯一ID和赛季信息
                cleaned_players['id'] = cleaned_players.index + len(data['players'])
                cleaned_players['season'] = season
                players_raw['season'] = season
                
                # 处理team信息
                if 'team' in players_raw.columns:
                    cleaned_players['team'] = players_raw['team']
                elif 'team_code' in players_raw.columns:
                    team_map = {
                        1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford',
                        5: 'Brighton', 6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton',
                        9: 'Fulham', 10: 'Leicester', 11: 'Leeds', 12: 'Liverpool',
                        13: 'Man City', 14: 'Man Utd', 15: 'Newcastle', 16: "Nott'm Forest",
                        17: 'Southampton', 18: 'Spurs', 19: 'West Ham', 20: 'Wolves'
                    }
                    cleaned_players['team'] = players_raw['team_code'].map(team_map)
                
                # 处理value信息
                if 'now_cost' in players_raw.columns:
                    cleaned_players['value'] = players_raw['now_cost'] / 10
                elif 'value' in merged_gw.columns:
                    value_data = merged_gw.groupby('name')['value'].mean().reset_index()
                    cleaned_players = cleaned_players.merge(value_data, on='name', how='left')
                
                # 确保所有必要的列都存在
                required_columns = ['id', 'name', 'position', 'team', 'value', 'total_points', 'minutes']
                missing_columns = [col for col in required_columns if col not in cleaned_players.columns]
                
                if missing_columns:
                    print(f"Warning: Missing columns for season {season}: {missing_columns}")
                    print("Available columns:", cleaned_players.columns.tolist())
                    # 填充缺失列
                    for col in missing_columns:
                        if col == 'total_points' and 'points' in cleaned_players.columns:
                            cleaned_players['total_points'] = cleaned_players['points']
                        elif col == 'minutes' and 'minutes_played' in cleaned_players.columns:
                            cleaned_players['minutes'] = cleaned_players['minutes_played']
                        else:
                            cleaned_players[col] = 0
                
                # 保存数据
                data['players'].append(cleaned_players)
                data['players_raw'].append(players_raw)
                data['gw'].append(merged_gw)
                data['fixtures'][season] = merged_gw
                successful_seasons.append(season)
                print(f"Successfully loaded season {season}")
                
            except Exception as e:
                print(f"Error processing season {season}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not successful_seasons:
            raise ValueError("No seasons were successfully loaded")
        
        print(f"\nSuccessfully loaded {len(successful_seasons)} seasons: {successful_seasons}")
        
        # 合并所有数据
        data['players'] = pd.concat(data['players'], ignore_index=True)
        data['players_raw'] = pd.concat(data['players_raw'], ignore_index=True)
        data['gw'] = pd.concat(data['gw'], ignore_index=True)
        
        # 打印数据统计信息
        print("\nData Statistics:")
        print(f"Total players: {len(data['players'])}")
        print(f"Total gameweeks: {len(data['gw'])}")
        print("\nColumns in players DataFrame:")
        print(data['players'].columns.tolist())
        
        return data
        
    async def generate_training_batch(self, 
                                    historical_data: Dict,
                                    batch_size: int = 32) -> Dict:
        """生成训练批次"""
        # 1. 从历史数据中采样
        players_sample = historical_data['players'].sample(batch_size)
        
        # 2. 获取环境数据
        env_data = []
        weather_data = {}
        media_data = {}
        venue_data = {}
        
        # 创建所有任务
        tasks = []
        for _, player in players_sample.iterrows():
            player_id = str(player.get('id', '0'))
            player_name = str(player.get('name', 'Unknown'))
            team = str(player.get('team', 'Unknown'))
            season = str(player.get('season', ''))
            
            # 只对当前赛季的数据进行实时API请求
            is_current_season = (season == self.current_season)
            
            # 获取场地数据
            venue_info = self.venue_collector.get_venue_data(team)
            if venue_info:
                venue_data[player_id] = venue_info
                venue_impact = self.venue_collector.calculate_venue_impact(team)
                if venue_impact['impact_score'] > 0:
                    env_data.append(
                        EnvironmentalData(
                            source=NewsSource.VENUE,
                            timestamp=datetime.now(),
                            description=f"Venue impact at {venue_info['name']}",
                            severity=venue_impact['impact_score'],
                            confidence=0.9,
                            event_type='venue',
                            venue_data=venue_info
                        )
                    )
            
            # 获取天气数据（只对当前赛季实时获取）
            if is_current_season and team in self.weather_collector.STADIUM_LOCATIONS:
                weather_info = self.weather_collector.get_weather_data(team)
                if weather_info:
                    weather_data[player_id] = weather_info
                    env_data.append(
                        EnvironmentalData(
                            source=NewsSource.WEATHER,
                            timestamp=datetime.now(),
                            description=f"Weather at {team}'s stadium",
                            severity=self._compute_weather_severity(weather_info),
                            confidence=0.9,
                            event_type=self._determine_weather_type(weather_info),
                            weather_details=weather_info
                        )
                    )
            else:
                # 对历史数据使用平均天气数据
                weather_data[player_id] = self._get_historical_weather(team, season)
            
            # 获取媒体数据（只对当前赛季实时获取）
            if is_current_season:
                task = asyncio.create_task(
                    self.media_collector.get_media_data(
                        player_name, 
                        team, 
                        days_back=3,
                        use_cache_only=not is_current_season  # 历史数据只使用缓存
                    )
                )
                tasks.append((player_id, task))
            else:
                # 对历史数据使用静态情感得分
                media_data[player_id] = self._get_historical_media_sentiment(player_name, team, season)
        
        # 等待所有媒体数据任务完成
        if tasks:  # 只有当前赛季才有异步任务
            for player_id, task in tasks:
                try:
                    articles = await task
                    if articles:
                        media_data[player_id] = articles
                        media_impact = await self.media_collector.create_media_impact(
                            player_name=players_sample[players_sample['id'] == int(player_id)]['name'].iloc[0],
                            team=players_sample[players_sample['id'] == int(player_id)]['team'].iloc[0]
                        )
                        if media_impact:
                            env_data.append(media_impact)
                except Exception as e:
                    print(f"获取媒体数据时出错: {str(e)}")
                    continue
        
        # 3. 构建训练批次
        batch = {
            'player_ids': [str(pid) for pid in players_sample['id'].tolist()],
            'player_names': {str(pid): name for pid, name in zip(players_sample['id'], players_sample['name'])},
            'positions': {str(pid): pos for pid, pos in zip(players_sample['id'], players_sample['position'])},
            'stats': {
                str(player['id']): {
                    'form': float(player.get('form', 0)) if pd.notnull(player.get('form')) else 0,
                    'fitness': float(player.get('minutes', 0)) / 90,
                    'momentum': float(player.get('points_per_game', 0)) if pd.notnull(player.get('points_per_game')) else 0,
                    'potential': float(player.get('value', 0)) / 100 if pd.notnull(player.get('value')) else 0
                }
                for _, player in players_sample.iterrows()
            },
            'env_data': env_data,
            'weather_data': weather_data,
            'media_data': media_data,
            'venue_data': venue_data,
            'formation': '4-4-2',  # 默认阵型
            'captain_id': str(players_sample['id'].iloc[0]),
            'target_points': players_sample['total_points'].tolist()
        }
        
        return batch
        
    def _compute_weather_severity(self, weather_data: Dict) -> float:
        """计算天气严重程度"""
        severity = 0.0
        
        # 温度影响
        temp = weather_data['temperature']
        if temp < 5 or temp > 30:
            severity += 0.4
        elif temp < 10 or temp > 25:
            severity += 0.2
            
        # 降水影响
        severity += weather_data['precipitation'] * 0.3
        
        # 风速影响
        severity += weather_data['wind_speed'] * 0.3
        
        return min(1.0, severity)
        
    def _determine_weather_type(self, weather_data: Dict) -> str:
        """确定天气类型"""
        if weather_data['precipitation'] > 0.5:
            return 'Heavy_Rain'
        elif weather_data['precipitation'] > 0:
            return 'Light_Rain'
        elif weather_data['wind_speed'] > 0.5:
            return 'Windy'
        elif weather_data['temperature'] < 5:
            return 'Cold'
        elif weather_data['temperature'] > 30:
            return 'Hot'
        return 'Moderate'
        
    def _get_historical_weather(self, team: str, season: str) -> Dict:
        """获取历史天气数据（使用平均值）"""
        return {
            'temperature': 15.0,  # 平均温度
            'precipitation': 0.0,  # 无降水
            'wind_speed': 2.0,    # 平均风速
            'humidity': 70,       # 平均湿度
            'condition': 'Normal' # 正常天气
        }
        
    def _get_historical_media_sentiment(self, player_name: str, team: str, season: str) -> List[Dict]:
        """获取历史媒体情感数据（使用静态数据）"""
        return [{
            'title': f'Historical data for {player_name}',
            'description': f'Static sentiment data for {player_name} in {season}',
            'source': 'Historical Database',
            'sentiment': {
                'score': 0.0,      # 中性情感
                'confidence': 0.8,  # 较高置信度
            }
        }]