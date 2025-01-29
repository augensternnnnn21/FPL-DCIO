# dsn_fpl/src/core/data/collector.py

from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

@dataclass
class RawPlayerData:
    """原始球员数据"""
    player_id: int
    name: str
    team: str
    position: str
    price: float
    total_points: int
    minutes: List[int]
    points: List[int]
    form: float
    points_per_game: float
    selected_by_percent: float
    status: str  # 可用性状态：'a' - 可用, 'i' - 受伤, 's' - 停赛, etc.

class NewsSource(Enum):
    WEATHER = "weather"
    NEWS = "news"
    TWITTER = "twitter"
    INJURY = "injury"
    VENUE = "venue"          # 场地因素
    MEDIA_SENTIMENT = "media"  # 媒体情感
    TEAM_FORM = "team"       # 球队状态
    RIVALRY = "rivalry"      # 对手关系

@dataclass
class EnvironmentalData:
    """环境数据"""
    source: NewsSource
    timestamp: datetime
    description: str
    severity: float  # 0-1
    confidence: float  # 0-1
    event_type: str
    related_players: Optional[List[int]] = None
    # 新增字段
    venue_data: Optional[Dict] = None  # 场地相关信息：{'is_home': bool, 'pitch_condition': float, 'crowd_factor': float}
    weather_details: Optional[Dict] = None  # 详细天气：{'temperature': float, 'precipitation': float, 'wind_speed': float}
    media_analysis: Optional[Dict] = None  # 媒体分析：{'sentiment_score': float, 'coverage_intensity': float}
    team_stats: Optional[Dict] = None  # 球队数据：{'form': float, 'goals_scored': int, 'goals_conceded': int}
    rivalry_info: Optional[Dict] = None  # 对手关系：{'historical_wins': int, 'historical_losses': int, 'importance': float}

class FPLDataCollector:
    """FPL数据收集器"""
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
        self.position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
    def get_season_data(self, season: str) -> Dict[str, pd.DataFrame]:
        """获取某个赛季的数据"""
        try:
            # 使用latin-1编码读取数据
            players_raw = pd.read_csv(
                f"{self.base_url}/{season}/players_raw.csv",
                encoding='latin-1'
            )
            cleaned_players = pd.read_csv(
                f"{self.base_url}/{season}/cleaned_players.csv",
                encoding='latin-1'
            )
            merged_gw = pd.read_csv(
                f"{self.base_url}/{season}/gws/merged_gw.csv",
                encoding='latin-1'
            )
            
            # 处理位置信息
            players_raw['position'] = players_raw['element_type'].map(self.position_map)
            
            return {
                'players_raw': players_raw,
                'cleaned_players': cleaned_players,
                'merged_gw': merged_gw
            }
        except Exception as e:
            print(f"Error loading season {season} data: {e}")
            return {}
            
    def get_player_data(self, player_id: int, season_data: Dict[str, pd.DataFrame]) -> Optional[RawPlayerData]:
        """获取球员数据"""
        try:
            player_raw = season_data['players_raw'][season_data['players_raw']['id'] == player_id].iloc[0]
            player_clean = season_data['cleaned_players'][season_data['cleaned_players']['id'] == player_id].iloc[0]
            player_gw = season_data['merged_gw'][season_data['merged_gw']['id'] == player_id]
            
            return RawPlayerData(
                player_id=player_id,
                name=player_raw['name'],
                team=player_raw['team'],
                position=player_raw['position'],
                price=player_raw['now_cost'] / 10,
                total_points=player_clean['total_points'],
                minutes=player_gw['minutes'].tolist(),
                points=player_gw['total_points'].tolist(),
                form=float(player_raw['form']),
                points_per_game=float(player_raw['points_per_game']),
                selected_by_percent=float(player_raw['selected_by_percent']),
                status=player_raw.get('status', 'a')
            )
        except Exception as e:
            print(f"Error getting player data for ID {player_id}: {e}")
            return None
            
    def get_player_history(self, season: str, player_name: str) -> Optional[pd.DataFrame]:
        """获取球员历史数据"""
        try:
            history = pd.read_csv(
                f"{self.base_url}/{season}/players/{player_name}/history.csv"
            )
            return history
        except:
            return None
            
    def get_current_gameweek_data(self, season: str, gw: int) -> Optional[pd.DataFrame]:
        """获取当前轮次数据"""
        try:
            gw_data = pd.read_csv(
                f"{self.base_url}/{season}/gws/gw{gw}.csv"
            )
            return gw_data
        except:
            return None
            
    def create_environmental_data(self, 
                                player_data: pd.DataFrame) -> List[EnvironmentalData]:
        """创建环境数据"""
        env_data = []
        
        # 从球员数据中提取相关信息
        for _, player in player_data.iterrows():
            # 检查伤病情况
            if player.get('status') != 'a':  # 假设'a'表示可用
                env_data.append(
                    EnvironmentalData(
                        source=NewsSource.INJURY,
                        timestamp=datetime.now(),
                        description=f"Player {player['name']} is not available",
                        severity=1.0,
                        confidence=1.0,
                        event_type='injury',
                        related_players=[player['id']]
                    )
                )
            
            # 检查表现趋势
            if 'form' in player.columns:
                form = float(player['form']) if pd.notnull(player['form']) else 0
                if form > 8.0:  # 高光表现
                    env_data.append(
                        EnvironmentalData(
                            source=NewsSource.NEWS,
                            timestamp=datetime.now(),
                            description=f"Player {player['name']} in excellent form",
                            severity=0.8,
                            confidence=0.9,
                            event_type='form',
                            related_players=[player['id']]
                        )
                    )
                    
        return env_data