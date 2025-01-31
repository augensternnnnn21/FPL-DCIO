from typing import Dict, Optional
import requests
from datetime import datetime
from ..data.collector import EnvironmentalData, NewsSource
import time

class WeatherCollector:
    """天气数据收集器"""
    
    # 英超球场位置
    STADIUM_LOCATIONS = {
        'Arsenal': 'London',
        'Chelsea': 'London',
        'Tottenham': 'London',
        'West Ham': 'London',
        'Crystal Palace': 'London',
        'Brentford': 'London',
        'Manchester City': 'Manchester',
        'Manchester United': 'Manchester',
        'Liverpool': 'Liverpool',
        'Everton': 'Liverpool',
        'Newcastle': 'Newcastle upon Tyne',
        'Aston Villa': 'Birmingham',
        'Brighton': 'Brighton',
        'Burnley': 'Burnley',
        'Nottingham Forest': 'Nottingham',
        'Sheffield United': 'Sheffield',
        'Bournemouth': 'Bournemouth',
        'Fulham': 'London',
        'Luton': 'Luton',
        'Wolves': 'Wolverhampton'
    }

    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: int = 2):
        """初始化收集器
        
        Args:
            api_key: WeatherAPI.com的API密钥
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.api_key = api_key
        self.base_url = "http://api.weatherapi.com/v1"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_weather_data(self, team: str) -> Optional[Dict]:
        """获取指定球队主场的天气数据
        
        Args:
            team: 球队名称
            
        Returns:
            包含天气数据的字典，如果获取失败则返回None
        """
        if team not in self.STADIUM_LOCATIONS:
            return None
            
        location = self.STADIUM_LOCATIONS[team]
        
        for attempt in range(self.max_retries):
            try:
                # 构建API请求URL
                url = f"{self.base_url}/current.json"
                params = {
                    'key': self.api_key,
                    'q': location,
                    'aqi': 'no'  # 不需要空气质量数据
                }
                
                # 发送请求
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # 提取需要的数据
                current = data['current']
                return {
                    'temperature': current['temp_c'],
                    'precipitation': current['precip_mm'],
                    'wind_speed': current['wind_kph'] / 3.6,  # 转换为m/s
                    'humidity': current['humidity'],
                    'condition': current['condition']['text'],
                    'is_day': bool(current['is_day']),
                    'cloud': current['cloud'],
                    'timestamp': datetime.now().isoformat()
                }
                
            except requests.exceptions.RequestException as e:
                print(f"第{attempt + 1}次尝试获取天气数据失败: {str(e)}")
                if attempt < self.max_retries - 1:
                    print(f"等待{self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    print("已达到最大重试次数，获取天气数据失败")
                    return None
            except Exception as e:
                print(f"获取天气数据时出错: {str(e)}")
                return None
        
    def create_weather_impact(self, team: str, match_time: datetime) -> Optional[EnvironmentalData]:
        """创建天气环境影响数据"""
        weather_data = self.get_weather_data(team)
        if not weather_data:
            return None
            
        # 计算天气严重程度
        severity = self._compute_weather_severity(weather_data)
        
        return EnvironmentalData(
            source=NewsSource.WEATHER,
            timestamp=match_time,
            description=self._generate_weather_description(weather_data),
            severity=severity,
            confidence=0.9,  # 天气数据可信度较高
            event_type=self._determine_weather_type(weather_data),
            weather_details=weather_data
        )
        
    def _compute_weather_severity(self, weather_data: Dict) -> float:
        """计算天气影响的严重程度"""
        severity = 0.0
        
        # 温度影响 (最适宜温度15-25度)
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
        
    def _generate_weather_description(self, weather_data: Dict) -> str:
        """生成天气描述"""
        desc = f"Temperature: {weather_data['temperature']}°C, "
        if weather_data['precipitation'] > 0:
            desc += f"Rain: {weather_data['precipitation']}mm/h, "
        desc += f"Wind: {weather_data['wind_speed']*10}m/s"
        return desc
        
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

    async def get_historical_weather(self, team: str, date: datetime) -> Optional[Dict]:
        """获取历史天气数据
        
        Args:
            team: 球队名称
            date: 历史日期
            
        Returns:
            包含天气信息的字典，如果获取失败则返回None
        """
        if team not in self.STADIUM_LOCATIONS:
            print(f"未知球队: {team}")
            return None
            
        location = self.STADIUM_LOCATIONS[team]
        
        try:
            # 构建API URL
            date_str = date.strftime('%Y-%m-%d')
            url = f"http://api.weatherapi.com/v1/history.json?key={self.api_key}&q={location}&dt={date_str}"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    print(f"API请求失败: {response.status}")
                    return None
                    
                data = await response.json()
                
                # 提取天气数据
                day = data['forecast']['forecastday'][0]['day']
                return {
                    'temperature': day['avgtemp_c'],
                    'precipitation': day['totalprecip_mm'],
                    'wind_speed': day['maxwind_kph'],
                    'humidity': day['avghumidity'],
                    'condition': day['condition']['text']
                }
                
        except Exception as e:
            print(f"获取历史天气数据时出错: {str(e)}")
            return None 