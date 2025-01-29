import os
from dotenv import load_dotenv

class Config:
    """配置管理器"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """加载配置"""
        # 加载.env文件
        load_dotenv()

        # API密钥
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')

        # 环境设置
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        self.cache_enabled = os.getenv('CACHE_ENABLED', 'True').lower() == 'true'
        self.cache_ttl = int(os.getenv('CACHE_TTL', '3600'))

    @property
    def has_weather_api(self) -> bool:
        """检查是否配置了天气API"""
        return bool(self.weather_api_key)

    @property
    def has_news_api(self) -> bool:
        """检查是否配置了新闻API"""
        return bool(self.news_api_key) 