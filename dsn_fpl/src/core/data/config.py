import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent.parent.parent.parent / '.env'
load_dotenv(env_path)

class Config:
    def __init__(self):
        # API密钥
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        
        # 数据路径
        self.data_dir = Path(__file__).parent.parent.parent.parent / 'data'
        
        # 缓存配置
        self.cache_dir = Path(__file__).parent.parent.parent.parent / 'cache'
        self.cache_ttl = 3600  # 缓存过期时间（秒）
        
        # 训练配置
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 100
        
        # 场地配置
        self.venue_data = {
            'Arsenal': {
                'name': 'Emirates Stadium',
                'capacity': 60704,
                'pitch_size': (105, 68),
                'surface_type': 'grass',
                'location': 'London',
                'atmosphere_rating': 0.90,  # 球场氛围评分
                'average_attendance': 59708,  # 平均上座人数
                'season_ticket_holders': 45000,  # 季票持有者数量
                'away_allocation': 3000,  # 客队球迷席位
                'fan_noise_level': 0.85  # 球迷噪音水平
            },
            'Manchester City': {
                'name': 'Etihad Stadium',
                'capacity': 53400,
                'pitch_size': (105, 68),
                'surface_type': 'grass',
                'location': 'Manchester',
                'atmosphere_rating': 0.82,
                'average_attendance': 52000,
                'season_ticket_holders': 36000,
                'away_allocation': 3000,
                'fan_noise_level': 0.80
            },
            'Liverpool': {
                'name': 'Anfield',
                'capacity': 53394,
                'pitch_size': (101, 68),
                'surface_type': 'grass',
                'location': 'Liverpool',
                'atmosphere_rating': 0.95,
                'average_attendance': 52964,
                'season_ticket_holders': 27000,
                'away_allocation': 3000,
                'fan_noise_level': 0.95
            },
            'Chelsea': {
                'name': 'Stamford Bridge',
                'capacity': 40341,
                'pitch_size': (103, 67),
                'surface_type': 'grass',
                'location': 'London',
                'atmosphere_rating': 0.85,
                'average_attendance': 39000,
                'season_ticket_holders': 25000,
                'away_allocation': 3000,
                'fan_noise_level': 0.82
            },
            'Manchester United': {
                'name': 'Old Trafford',
                'capacity': 74140,
                'pitch_size': (105, 68),
                'surface_type': 'grass',
                'location': 'Manchester',
                'atmosphere_rating': 0.92,
                'average_attendance': 73000,
                'season_ticket_holders': 52000,
                'away_allocation': 3000,
                'fan_noise_level': 0.90
            },
            'Tottenham': {
                'name': 'Tottenham Hotspur Stadium',
                'capacity': 62850,
                'pitch_size': (105, 68),
                'surface_type': 'grass',
                'location': 'London',
                'atmosphere_rating': 0.88,
                'average_attendance': 61000,
                'season_ticket_holders': 42000,
                'away_allocation': 3000,
                'fan_noise_level': 0.88
            }
        }
        
        # 球迷影响力配置
        self.fan_impact_weights = {
            'atmosphere_rating': 0.3,    # 球场氛围权重
            'attendance_ratio': 0.2,     # 上座率权重
            'season_ticket_ratio': 0.2,  # 季票持有率权重
            'noise_level': 0.3          # 噪音水平权重
        }
        
        self._validate_config()
    
    def _validate_config(self):
        """验证配置是否有效"""
        if not self.weather_api_key:
            raise ValueError("Weather API key not found in environment variables")
        if not self.news_api_key:
            raise ValueError("News API key not found in environment variables")
            
        # 确保数据目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True) 