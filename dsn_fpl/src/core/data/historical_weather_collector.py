import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import pytz
from bs4 import BeautifulSoup
import json
import logging
import ssl
import urllib3
from meteostat import Point, Daily, Hourly, Stations
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import random
import time
from .historical_data_manager import HistoricalDataManager

# 禁用SSL警告
urllib3.disable_warnings()

class HistoricalWeatherCollector:
    """历史天气数据收集器"""
    
    # 英超球场位置坐标（经纬度）
    STADIUM_LOCATIONS = {
        'Arsenal': {'lat': 51.5549, 'lon': -0.1084},
        'Chelsea': {'lat': 51.4817, 'lon': -0.1910},
        'Tottenham': {'lat': 51.6033, 'lon': -0.0659},
        'West Ham': {'lat': 51.5387, 'lon': -0.0166},
        'Crystal Palace': {'lat': 51.3983, 'lon': -0.0866},
        'Brentford': {'lat': 51.4882, 'lon': -0.2887},
        'Manchester City': {'lat': 53.4831, 'lon': -2.2004},
        'Manchester United': {'lat': 53.4631, 'lon': -2.2913},
        'Liverpool': {'lat': 53.4308, 'lon': -2.9608},
        'Everton': {'lat': 53.4389, 'lon': -2.9664},
        'Newcastle': {'lat': 54.9756, 'lon': -1.6217},
        'Aston Villa': {'lat': 52.5090, 'lon': -1.8845},
        'Brighton': {'lat': 50.8614, 'lon': -0.0831},
        'Burnley': {'lat': 53.7890, 'lon': -2.2302},
        'Nottingham Forest': {'lat': 52.9400, 'lon': -1.1325},
        'Sheffield United': {'lat': 53.3703, 'lon': -1.4707},
        'Bournemouth': {'lat': 50.7352, 'lon': -1.8385},
        'Fulham': {'lat': 51.4750, 'lon': -0.2219},
        'Luton': {'lat': 51.8843, 'lon': -0.4347},
        'Wolves': {'lat': 52.5900, 'lon': -2.1300}
    }

    def __init__(self, data_path: Optional[Path] = None):
        """初始化收集器
        
        Args:
            data_path: 数据根目录路径
        """
        self.data_path = data_path or Path.cwd() / "data"
        self.historical_path = self.data_path / "historical" / "weather"
        self.historical_path.mkdir(parents=True, exist_ok=True)
        
        self.data_manager = HistoricalDataManager(self.data_path)
        self.cache = {}
        self.user_agent = UserAgent()
        self._request_semaphore = asyncio.Semaphore(3)  # 减少并发请求数
        self._request_interval = (3, 8)  # 增加延时范围
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 创建SSL上下文
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
    def _parse_datetime(self, dt_str: str) -> datetime:
        """解析日期时间字符串
        
        Args:
            dt_str: 日期时间字符串，格式如 "2023-08-12T14:00:00Z"
            
        Returns:
            datetime: 时区感知的datetime对象
        """
        try:
            # 直接解析带有Z后缀的UTC时间
            dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
            return dt.replace(tzinfo=pytz.UTC)
        except ValueError as e:
            # 如果解析失败，尝试其他格式
            try:
                dt = pd.to_datetime(dt_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pytz.UTC)
                return dt.astimezone(pytz.UTC)
            except Exception as e:
                logging.error(f"无法解析日期时间: {dt_str}, 错误: {str(e)}")
                raise
        
    async def collect_historical_weather(self, season: str):
        """收集指定赛季的历史天气数据
        
        Args:
            season: 赛季，格式如 "2022-23"
        """
        logging.info(f"开始收集 {season} 赛季的天气数据...")
        
        # 读取赛季比赛数据
        fixtures_path = self.data_path / season / "fixtures.csv"
        if not fixtures_path.exists():
            logging.error(f"找不到赛季 {season} 的比赛数据")
            return
            
        try:
            fixtures_df = pd.read_csv(fixtures_path)
            teams_df = pd.read_csv(self.data_path / season / "teams.csv")
            
            # 按球队分组处理
            team_fixtures = self._group_fixtures_by_team(fixtures_df, teams_df)
            
            # 并发处理每支球队的数据
            tasks = []
            for team_name, team_data in team_fixtures.items():
                task = self._process_team_fixtures(team_name, team_data)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计结果
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            logging.info(f"完成 {season} 赛季数据收集，成功处理 {success_count} 支球队")
            
            # 保存缓存数据
            self._save_cache(season)
            
        except Exception as e:
            logging.error(f"处理赛季 {season} 数据时出错: {str(e)}")
            
    def _group_fixtures_by_team(self, fixtures_df: pd.DataFrame, teams_df: pd.DataFrame) -> Dict:
        """将比赛按主场球队分组"""
        team_fixtures = {}
        
        # 创建team_id到team_name的映射
        team_map = dict(zip(teams_df['team_id'], teams_df['name']))
        
        for _, fixture in fixtures_df.iterrows():
            home_team = team_map.get(fixture['team_h'])
            if home_team and home_team in self.STADIUM_LOCATIONS:
                if home_team not in team_fixtures:
                    team_fixtures[home_team] = []
                team_fixtures[home_team].append({
                    'date': fixture['kickoff_time'],
                    'opponent': team_map.get(fixture['team_a'], 'Unknown'),
                    'score': f"{fixture['team_h_score']}-{fixture['team_a_score']}"
                })
                
        return team_fixtures
        
    async def _process_team_fixtures(self, team_name: str, fixtures: List[Dict]):
        """处理单个球队的所有比赛"""
        logging.info(f"处理 {team_name} 的比赛数据...")
        
        weather_data = []
        for fixture in fixtures:
            try:
                # 使用新的解析方法
                match_time = self._parse_datetime(fixture['date'])
                weather = await self._get_match_weather(team_name, match_time)
                
                if weather:
                    weather.update({
                        'team': team_name,
                        'opponent': fixture['opponent'],
                        'score': fixture['score'],
                        'match_time': match_time.isoformat()
                    })
                    weather_data.append(weather)
                    
            except Exception as e:
                logging.error(f"处理 {team_name} vs {fixture['opponent']} 的天气数据时出错: {str(e)}")
                
        return weather_data
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception_type((IOError, ssl.SSLError, ConnectionError))
    )
    async def _get_match_weather(self, team: str, match_time: datetime) -> Optional[Dict]:
        """获取单场比赛的天气数据"""
        # 首先尝试使用备用爬虫方案
        try:
            weather = await self._get_backup_weather(team, match_time)
            if weather:
                return weather
        except Exception as e:
            logging.warning(f"备用数据源获取失败: {str(e)}")
            
        # 如果备用方案失败，尝试使用Meteostat
        try:
            coords = self.STADIUM_LOCATIONS[team]
            point = Point(coords['lat'], coords['lon'])
            
            # 确保使用naive datetime，因为Meteostat内部会处理时区
            start_time = match_time.replace(tzinfo=None)
            end_time = (start_time + timedelta(hours=1))
            
            # 获取比赛当天的小时数据
            data = Hourly(point, start_time, end_time)
            data = data.fetch()
            
            if not data.empty:
                row = data.iloc[0]
                return {
                    'temperature': row.get('temp', 15.0),
                    'precipitation': row.get('prcp', 0.0),
                    'wind_speed': row.get('wdir', 0.0),
                    'humidity': row.get('rhum', 70),
                    'condition': self._determine_condition(row),
                    'data_source': 'meteostat'
                }
        except Exception as e:
            logging.warning(f"Meteostat数据获取失败: {str(e)}")
            
        return None
        
    async def _get_backup_weather(self, team: str, match_time: datetime) -> Optional[Dict]:
        """备用天气数据获取方案"""
        # 随机延时，避免请求过于频繁
        await asyncio.sleep(random.uniform(*self._request_interval))
        
        async with self._request_semaphore:
            headers = {
                'User-Agent': self.user_agent.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            
            # 尝试从visualcrossing.com获取数据
            coords = self.STADIUM_LOCATIONS[team]
            date_str = match_time.strftime('%Y-%m-%d')
            url = f"https://www.visualcrossing.com/weather-history/{coords['lat']},{coords['lon']}/{date_str}"
            
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url, ssl=self.ssl_context) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # 解析天气数据
                            weather_div = soup.find('div', {'class': 'weather-historic'})
                            if weather_div:
                                return {
                                    'temperature': self._extract_temperature(weather_div),
                                    'precipitation': self._extract_precipitation(weather_div),
                                    'wind_speed': self._extract_wind_speed(weather_div),
                                    'humidity': self._extract_humidity(weather_div),
                                    'condition': self._extract_condition(weather_div),
                                    'data_source': 'visualcrossing'
                                }
                                
            except Exception as e:
                logging.warning(f"Visual Crossing数据获取失败: {str(e)}")
                
            # 如果Visual Crossing失败，尝试从timeanddate.com获取数据
            try:
                url = f"https://www.timeanddate.com/weather/@{coords['lat']},{coords['lon']}/historic?month={match_time.month}&year={match_time.year}"
                
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url, ssl=self.ssl_context) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # 找到目标日期的天气数据
                            target_date = match_time.strftime('%Y-%m-%d')
                            weather_table = soup.find('table', {'class': 'zebra'})
                            
                            if weather_table:
                                for row in weather_table.find_all('tr')[1:]:  # 跳过表头
                                    cols = row.find_all('td')
                                    if len(cols) >= 7:
                                        date_col = cols[0].text.strip()
                                        if target_date in date_col:
                                            return {
                                                'temperature': float(cols[2].text.strip().replace('°C', '')),
                                                'precipitation': float(cols[4].text.strip().replace('mm', '') or 0),
                                                'wind_speed': float(cols[6].text.strip().split()[0]),
                                                'humidity': int(cols[3].text.strip().replace('%', '')),
                                                'condition': cols[1].text.strip(),
                                                'data_source': 'timeanddate'
                                            }
                                            
            except Exception as e:
                logging.warning(f"Time and Date数据获取失败: {str(e)}")
                
        return None
        
    def _determine_condition(self, row) -> str:
        """根据天气参数确定天气状况"""
        if row.get('prcp', 0) > 0:
            if row.get('prcp') > 5:
                return 'Heavy Rain'
            return 'Light Rain'
        elif row.get('temp', 15) < 5:
            return 'Cold'
        elif row.get('temp', 15) > 25:
            return 'Hot'
        return 'Normal'
        
    def _extract_temperature(self, weather_div) -> float:
        """从HTML中提取温度数据"""
        try:
            temp_elem = weather_div.find('span', {'class': 'temp'})
            if temp_elem:
                return float(temp_elem.text.strip().replace('°C', ''))
        except:
            pass
        return 15.0
        
    def _extract_precipitation(self, weather_div) -> float:
        """从HTML中提取降水数据"""
        try:
            precip_elem = weather_div.find('span', {'class': 'precip'})
            if precip_elem:
                return float(precip_elem.text.strip().replace('mm', ''))
        except:
            pass
        return 0.0
        
    def _extract_wind_speed(self, weather_div) -> float:
        """从HTML中提取风速数据"""
        try:
            wind_elem = weather_div.find('span', {'class': 'wind'})
            if wind_elem:
                return float(wind_elem.text.strip().split()[0])
        except:
            pass
        return 2.0
        
    def _extract_humidity(self, weather_div) -> int:
        """从HTML中提取湿度数据"""
        try:
            humidity_elem = weather_div.find('span', {'class': 'humidity'})
            if humidity_elem:
                return int(humidity_elem.text.strip().replace('%', ''))
        except:
            pass
        return 70
        
    def _extract_condition(self, weather_div) -> str:
        """从HTML中提取天气状况"""
        try:
            condition_elem = weather_div.find('span', {'class': 'condition'})
            if condition_elem:
                return condition_elem.text.strip()
        except:
            pass
        return 'Normal'
        
    def _save_cache(self, season: str):
        """保存缓存数据到文件"""
        if self.cache:
            cache_file = self.historical_path / f"{season}_weather_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f, indent=4)
            logging.info(f"保存缓存数据: {len(self.cache)} 条记录")
            
    async def close(self):
        """关闭收集器"""
        pass 