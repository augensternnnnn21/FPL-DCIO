import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

class HistoricalDataManager:
    """历史数据管理器，用于处理历史天气和媒体数据"""
    
    def __init__(self, data_path: Optional[Path] = None):
        """初始化历史数据管理器
        
        Args:
            data_path: 数据根目录路径，默认为项目根目录下的data文件夹
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent.parent.parent / "data"
        self.data_path = Path(data_path)
        self.historical_path = self.data_path / "historical"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config_path = self.historical_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
            
    def get_weather_data(self, season: str, team: str, match_date: Optional[datetime] = None) -> Dict:
        """获取历史天气数据
        
        Args:
            season: 赛季，格式如 "2022-23"
            team: 球队名称
            match_date: 比赛日期（可选）
            
        Returns:
            包含天气信息的字典
        """
        file_path = self.historical_path / "weather" / f"{season}_{team}_weather.csv"
        if not file_path.exists():
            return self._get_default_weather()
            
        df = pd.read_csv(file_path)
        if match_date:
            # 如果指定了日期，返回最接近的天气记录
            df['match_date'] = pd.to_datetime(df['match_date'])
            closest_date = df.iloc[(df['match_date'] - match_date).abs().argsort()[0]]
            return closest_date.to_dict()
        else:
            # 否则返回平均天气数据
            return {
                'temperature': df['temperature'].mean(),
                'precipitation': df['precipitation'].mean(),
                'wind_speed': df['wind_speed'].mean(),
                'humidity': df['humidity'].mean(),
                'condition': df['condition'].mode()[0]
            }
            
    def get_media_data(self, season: str, player_name: str) -> List[Dict]:
        """获取历史媒体数据
        
        Args:
            season: 赛季，格式如 "2022-23"
            player_name: 球员名称
            
        Returns:
            包含媒体文章的列表
        """
        file_path = self.historical_path / "media" / f"{season}_{player_name}_media.json"
        if not file_path.exists():
            return self._get_default_media(player_name)
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('articles', [])
            
    def save_weather_data(self, season: str, team: str, weather_data: Union[Dict, List[Dict]]):
        """保存天气数据
        
        Args:
            season: 赛季
            team: 球队名称
            weather_data: 天气数据（单条记录或记录列表）
        """
        file_path = self.historical_path / "weather" / f"{season}_{team}_weather.csv"
        if isinstance(weather_data, dict):
            weather_data = [weather_data]
            
        df = pd.DataFrame(weather_data)
        df.to_csv(file_path, index=False)
        
    def save_media_data(self, season: str, player_name: str, team: str, articles: List[Dict]):
        """保存媒体数据
        
        Args:
            season: 赛季
            player_name: 球员名称
            team: 球队名称
            articles: 文章列表
        """
        file_path = self.historical_path / "media" / f"{season}_{player_name}_media.json"
        data = {
            'player_name': player_name,
            'team': team,
            'season': season,
            'articles': articles
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def _get_default_weather(self) -> Dict:
        """获取默认天气数据"""
        return {
            'temperature': 15.0,
            'precipitation': 0.0,
            'wind_speed': 2.0,
            'humidity': 70,
            'condition': 'Normal'
        }
        
    def _get_default_media(self, player_name: str) -> List[Dict]:
        """获取默认媒体数据"""
        return [{
            'date': datetime.now().isoformat(),
            'title': f'Default article for {player_name}',
            'source': 'Historical Database',
            'sentiment_score': 0.0,
            'sentiment_confidence': 0.8
        }] 