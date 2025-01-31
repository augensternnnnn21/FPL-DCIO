import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import pytz
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import re
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import ssl
import aiohttp.client_exceptions

class NewsCollector:
    """新闻数据采集器"""
    
    def __init__(self, data_path: Optional[Path] = None):
        """初始化收集器
        
        Args:
            data_path: 数据存储路径
        """
        self.data_path = data_path or Path.cwd() / "data"
        self.news_path = self.data_path / "historical" / "news"
        self.news_path.mkdir(parents=True, exist_ok=True)
        
        # 配置请求头
        self.user_agent = UserAgent()
        self.headers = {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # 配置新闻源
        self.sources = {
            'bbc_sport': {
                'url': 'https://www.bbc.com/sport/football/teams/{team}/scores-fixtures',
                'team_map': {
                    'Arsenal': 'arsenal',
                    'Chelsea': 'chelsea',
                    'Tottenham': 'tottenham-hotspur',
                    'West Ham': 'west-ham-united',
                    'Crystal Palace': 'crystal-palace',
                    'Brentford': 'brentford',
                    'Manchester City': 'manchester-city',
                    'Manchester United': 'manchester-united',
                    'Liverpool': 'liverpool',
                    'Everton': 'everton',
                    'Newcastle': 'newcastle-united',
                    'Aston Villa': 'aston-villa',
                    'Brighton': 'brighton-and-hove-albion',
                    'Burnley': 'burnley',
                    'Nottingham Forest': 'nottingham-forest',
                    'Sheffield United': 'sheffield-united',
                    'Bournemouth': 'afc-bournemouth',
                    'Fulham': 'fulham',
                    'Luton': 'luton-town',
                    'Wolves': 'wolverhampton-wanderers'
                }
            },
            'sky_sports': {
                'url': 'https://www.skysports.com/{team}-news',
                'team_map': {
                    'Arsenal': 'arsenal',
                    'Chelsea': 'chelsea',
                    'Tottenham': 'tottenham-hotspur',
                    'West Ham': 'west-ham-united',
                    'Crystal Palace': 'crystal-palace',
                    'Brentford': 'brentford',
                    'Manchester City': 'manchester-city',
                    'Manchester United': 'manchester-united',
                    'Liverpool': 'liverpool',
                    'Everton': 'everton',
                    'Newcastle': 'newcastle-united',
                    'Aston Villa': 'aston-villa',
                    'Brighton': 'brighton',
                    'Burnley': 'burnley',
                    'Nottingham Forest': 'nottingham-forest',
                    'Sheffield United': 'sheffield-united',
                    'Bournemouth': 'bournemouth',
                    'Fulham': 'fulham',
                    'Luton': 'luton-town',
                    'Wolves': 'wolves'
                }
            }
        }
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 配置SSL上下文
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # 配置重试策略
        self.max_retries = 3
        self.retry_delay = (3, 10)  # 随机延时范围（秒）
        self.timeout = aiohttp.ClientTimeout(total=30)
        
    async def collect_match_news(self, team: str, match_time: datetime, opponent: str) -> Dict:
        """收集比赛相关的新闻
        
        Args:
            team: 球队名称
            match_time: 比赛时间
            opponent: 对手名称
            
        Returns:
            Dict: 包含比赛前后新闻的字典
        """
        # 设置时间范围
        pre_match_start = match_time - timedelta(days=2)
        pre_match_end = match_time
        post_match_start = match_time
        post_match_end = match_time + timedelta(days=1)
        
        # 收集新闻
        pre_match_news = await self._collect_news_in_timerange(
            team, opponent, pre_match_start, pre_match_end, 'pre_match'
        )
        post_match_news = await self._collect_news_in_timerange(
            team, opponent, post_match_start, post_match_end, 'post_match'
        )
        
        # 提取天气信息
        weather_info = self._extract_weather_info(pre_match_news + post_match_news)
        
        return {
            'team': team,
            'opponent': opponent,
            'match_time': match_time.isoformat(),
            'pre_match_news': pre_match_news,
            'post_match_news': post_match_news,
            'weather_info': weather_info
        }
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ssl.SSLError,
            ConnectionError
        ))
    )
    async def _fetch_url(self, url: str) -> str:
        """获取URL内容
        
        Args:
            url: 目标URL
            
        Returns:
            str: 网页内容
        """
        # 随机延时
        await asyncio.sleep(random.uniform(*self.retry_delay))
        
        # 更新User-Agent
        self.headers['User-Agent'] = self.user_agent.random
        
        async with aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout
        ) as session:
            async with session.get(
                url,
                ssl=self.ssl_context,
                allow_redirects=True
            ) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    raise aiohttp.ClientError(
                        f"HTTP {response.status}: {response.reason}"
                    )
                    
    async def _collect_news_in_timerange(
        self, team: str, opponent: str, start_time: datetime, end_time: datetime, news_type: str
    ) -> List[Dict]:
        """在指定时间范围内收集新闻
        
        Args:
            team: 球队名称
            opponent: 对手名称
            start_time: 开始时间
            end_time: 结束时间
            news_type: 新闻类型 ('pre_match' 或 'post_match')
            
        Returns:
            List[Dict]: 新闻列表
        """
        news = []
        
        # 从各个新闻源收集数据
        for source_name, source_info in self.sources.items():
            try:
                team_slug = source_info['team_map'].get(team)
                if not team_slug:
                    continue
                    
                url = source_info['url'].format(team=team_slug)
                html = await self._fetch_url(url)
                
                if source_name == 'bbc_sport':
                    news.extend(await self._parse_bbc_news(
                        html, team, opponent, start_time, end_time, news_type
                    ))
                elif source_name == 'sky_sports':
                    news.extend(await self._parse_sky_news(
                        html, team, opponent, start_time, end_time, news_type
                    ))
                    
            except Exception as e:
                logging.warning(f"从{source_name}获取新闻失败: {str(e)}")
                continue
                
        return news
        
    async def _parse_bbc_news(
        self, html: str, team: str, opponent: str, start_time: datetime,
        end_time: datetime, news_type: str
    ) -> List[Dict]:
        """解析BBC体育新闻
        
        Args:
            html: 网页内容
            team: 球队名称
            opponent: 对手名称
            start_time: 开始时间
            end_time: 结束时间
            news_type: 新闻类型
            
        Returns:
            List[Dict]: 新闻列表
        """
        news = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # 查找新闻文章
        articles = soup.find_all(['article', 'div'], class_=['gs-c-promo', 'gel-layout__item'])
        
        for article in articles:
            try:
                # 提取时间
                time_elem = article.find('time')
                if not time_elem or 'datetime' not in time_elem.attrs:
                    continue
                    
                pub_time = datetime.fromisoformat(time_elem['datetime']).replace(tzinfo=pytz.UTC)
                
                # 检查时间范围
                if not (start_time <= pub_time <= end_time):
                    continue
                    
                # 提取标题和链接
                title_elem = article.find(['h3', 'h2', 'h4'])
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                
                # 检查是否与比赛相关
                if not (team.lower() in title.lower() or opponent.lower() in title.lower()):
                    continue
                    
                # 提取链接和描述
                link = article.find('a')['href']
                if not link.startswith('http'):
                    link = f"https://www.bbc.com{link}"
                    
                desc_elem = article.find('p', class_=['gs-c-promo-summary', 'gel-body-copy'])
                description = desc_elem.text.strip() if desc_elem else ''
                
                news.append({
                    'title': title,
                    'description': description,
                    'url': link,
                    'published_at': pub_time.isoformat(),
                    'source': 'bbc_sport',
                    'type': news_type
                })
                
            except Exception as e:
                logging.warning(f"解析BBC新闻文章时出错: {str(e)}")
                continue
                
        return news
        
    async def _parse_sky_news(
        self, html: str, team: str, opponent: str, start_time: datetime,
        end_time: datetime, news_type: str
    ) -> List[Dict]:
        """解析Sky Sports新闻
        
        Args:
            html: 网页内容
            team: 球队名称
            opponent: 对手名称
            start_time: 开始时间
            end_time: 结束时间
            news_type: 新闻类型
            
        Returns:
            List[Dict]: 新闻列表
        """
        news = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # 查找新闻文章
        articles = soup.find_all(['div', 'article'], class_=['news-list__item', 'news-article'])
        
        for article in articles:
            try:
                # 提取时间
                time_elem = article.find('time')
                if not time_elem:
                    # 尝试从其他元素中提取时间
                    time_elem = article.find(['span', 'div'], class_=['news-date', 'article-date'])
                    if not time_elem:
                        continue
                        
                # 解析时间
                try:
                    if 'datetime' in time_elem.attrs:
                        pub_time = datetime.fromisoformat(time_elem['datetime']).replace(tzinfo=pytz.UTC)
                    else:
                        # 尝试从文本内容解析时间
                        time_text = time_elem.text.strip()
                        pub_time = datetime.strptime(time_text, '%d %B %Y %H:%M').replace(tzinfo=pytz.UTC)
                except:
                    continue
                    
                # 检查时间范围
                if not (start_time <= pub_time <= end_time):
                    continue
                    
                # 提取标题和链接
                title_elem = article.find(['h4', 'h2', 'h3'])
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                
                # 检查是否与比赛相关
                if not (team.lower() in title.lower() or opponent.lower() in title.lower()):
                    continue
                    
                # 提取链接和描述
                link = article.find('a')['href']
                if not link.startswith('http'):
                    link = f"https://www.skysports.com{link}"
                    
                desc_elem = article.find(['p', 'div'], class_=['news-summary', 'article-summary'])
                description = desc_elem.text.strip() if desc_elem else ''
                
                news.append({
                    'title': title,
                    'description': description,
                    'url': link,
                    'published_at': pub_time.isoformat(),
                    'source': 'sky_sports',
                    'type': news_type
                })
                
            except Exception as e:
                logging.warning(f"解析Sky Sports新闻文章时出错: {str(e)}")
                continue
                
        return news
        
    def _extract_weather_info(self, news_list: List[Dict]) -> Optional[Dict]:
        """从新闻中提取天气信息
        
        Args:
            news_list: 新闻列表
            
        Returns:
            Optional[Dict]: 天气信息字典
        """
        weather_info = None
        
        # 天气相关的关键词模式
        temp_pattern = r'(\d+)°C|(\d+) degrees'
        rain_pattern = r'rain|shower|drizzle|precipitation'
        wind_pattern = r'(\d+)\s*mph|(\d+)\s*km/h'
        
        for news in news_list:
            text = f"{news['title']} {news['description']}"
            
            # 提取温度
            temp_match = re.search(temp_pattern, text)
            if temp_match:
                if not weather_info:
                    weather_info = {}
                weather_info['temperature'] = float(
                    temp_match.group(1) or temp_match.group(2)
                )
                
            # 检查降水情况
            if re.search(rain_pattern, text.lower()):
                if not weather_info:
                    weather_info = {}
                weather_info['precipitation'] = True
                
            # 提取风速
            wind_match = re.search(wind_pattern, text)
            if wind_match:
                if not weather_info:
                    weather_info = {}
                wind_speed = float(wind_match.group(1) or wind_match.group(2))
                if 'mph' in text:
                    wind_speed *= 1.60934  # 转换为km/h
                weather_info['wind_speed'] = wind_speed
                
        return weather_info
        
    def save_news(self, season: str, news_data: Dict):
        """保存新闻数据
        
        Args:
            season: 赛季
            news_data: 新闻数据
        """
        season_path = self.news_path / season
        season_path.mkdir(exist_ok=True)
        
        # 使用比赛时间和球队名称作为文件名
        match_time = datetime.fromisoformat(news_data['match_time'])
        filename = f"{match_time.strftime('%Y%m%d')}_{news_data['team']}_{news_data['opponent']}.json"
        
        with open(season_path / filename, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"保存新闻数据: {filename}") 