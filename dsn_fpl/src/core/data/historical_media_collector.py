import asyncio
import aiohttp
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from .historical_data_manager import HistoricalDataManager
from .media_collector import MediaCollector
from .sentiment_analyzer import SentimentAnalyzer

class HistoricalMediaCollector:
    """历史新闻数据收集器"""
    
    GUARDIAN_API_BASE = "https://content.guardianapis.com/search"
    WEB_ARCHIVE_BASE = "https://archive.org/wayback/available"
    BBC_ARCHIVE_BASE = "https://www.bbc.co.uk/sport/football"
    
    def __init__(self, 
                 guardian_api_key: str,
                 news_api_key: str,
                 data_path: Optional[Path] = None):
        """初始化收集器
        
        Args:
            guardian_api_key: Guardian API密钥
            news_api_key: NewsAPI密钥
            data_path: 数据根目录路径
        """
        self.guardian_api_key = guardian_api_key
        self.media_collector = MediaCollector(news_api_key=news_api_key)
        self.data_manager = HistoricalDataManager(data_path)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.session = None
        self._request_semaphore = asyncio.Semaphore(1)
        self._request_interval = 2.0
        
    async def collect_historical_media(self, season: str):
        """收集指定赛季的历史新闻数据
        
        Args:
            season: 赛季，格式如 "2022-23"
        """
        # 读取球员数据
        players_path = self.data_manager.data_path / season / "cleaned_players.csv"
        if not players_path.exists():
            print(f"找不到赛季 {season} 的球员数据")
            return
            
        players_df = pd.read_csv(players_path)
        
        # 确保session已创建
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
        # 获取赛季的开始和结束日期
        season_dates = self._get_season_dates(season)
        
        # 按球员收集新闻数据
        for _, player in players_df.iterrows():
            player_name = player['name']
            team = player['team']
            print(f"\n收集 {player_name} ({team}) 的新闻数据...")
            
            articles = []
            seen_urls = set()  # 用于去重
            
            # 1. 从Guardian获取数据
            try:
                guardian_articles = await self._get_guardian_articles(
                    player_name,
                    team,
                    season_dates['start'],
                    season_dates['end']
                )
                articles.extend(self._deduplicate_articles(guardian_articles, seen_urls))
                print(f"从Guardian获取到 {len(guardian_articles)} 篇文章")
            except Exception as e:
                print(f"获取Guardian数据时出错: {str(e)}")
                
            # 2. 从BBC Sport Archive获取数据
            try:
                bbc_articles = await self._get_bbc_archive_articles(
                    player_name,
                    team,
                    season_dates['start'],
                    season_dates['end']
                )
                articles.extend(self._deduplicate_articles(bbc_articles, seen_urls))
                print(f"从BBC获取到 {len(bbc_articles)} 篇文章")
            except Exception as e:
                print(f"获取BBC数据时出错: {str(e)}")
                
            # 3. 从Web Archive获取数据
            try:
                archive_articles = await self._get_web_archive_articles(
                    player_name,
                    team,
                    season_dates['start'],
                    season_dates['end']
                )
                articles.extend(self._deduplicate_articles(archive_articles, seen_urls))
                print(f"从Web Archive获取到 {len(archive_articles)} 篇文章")
            except Exception as e:
                print(f"获取Web Archive数据时出错: {str(e)}")
                
            # 分析情感并保存数据
            if articles:
                # 对每篇文章进行情感分析
                for article in articles:
                    sentiment = self.sentiment_analyzer.analyze_text(
                        f"{article['title']} {article.get('description', '')}"
                    )
                    article['sentiment'] = sentiment
                
                # 保存数据
                self.data_manager.save_media_data(season, player_name, team, articles)
                print(f"已保存 {len(articles)} 篇关于 {player_name} 的新闻")
            else:
                print(f"未找到关于 {player_name} 的新闻")
                
    async def _get_guardian_articles(self, 
                                   player: str, 
                                   team: str, 
                                   start_date: datetime,
                                   end_date: datetime) -> List[Dict]:
        """从Guardian API获取历史文章"""
        articles = []
        async with self._request_semaphore:
            await asyncio.sleep(self._request_interval)
            
            params = {
                'api-key': self.guardian_api_key,
                'q': f'"{player}" "{team}"',
                'section': 'football',
                'from-date': start_date.strftime('%Y-%m-%d'),
                'to-date': end_date.strftime('%Y-%m-%d'),
                'page-size': 50
            }
            
            try:
                async with self.session.get(self.GUARDIAN_API_BASE, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for result in data['response']['results']:
                            articles.append({
                                'date': result['webPublicationDate'],
                                'title': result['webTitle'],
                                'source': 'The Guardian',
                                'url': result['webUrl']
                            })
            except Exception as e:
                print(f"Guardian API请求失败: {str(e)}")
                
        return articles
        
    async def _get_bbc_archive_articles(self,
                                      player: str,
                                      team: str,
                                      start_date: datetime,
                                      end_date: datetime) -> List[Dict]:
        """从BBC Sport Archive获取历史文章"""
        articles = []
        async with self._request_semaphore:
            await asyncio.sleep(self._request_interval)
            
            # 构建搜索URL
            search_url = f"{self.BBC_ARCHIVE_BASE}/search?q={player}+{team}"
            
            try:
                async with self.session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # 解析搜索结果
                        for article in soup.select('.gs-c-promo'):
                            date_elem = article.select_one('.gs-c-promo-date')
                            if date_elem:
                                pub_date = datetime.strptime(
                                    date_elem['datetime'],
                                    '%Y-%m-%d'
                                )
                                
                                if start_date <= pub_date <= end_date:
                                    articles.append({
                                        'date': pub_date.isoformat(),
                                        'title': article.select_one('.gs-c-promo-heading').text.strip(),
                                        'source': 'BBC Sport',
                                        'url': article.select_one('a')['href']
                                    })
            except Exception as e:
                print(f"BBC Archive请求失败: {str(e)}")
                
        return articles
        
    async def _get_web_archive_articles(self,
                                      player: str,
                                      team: str,
                                      start_date: datetime,
                                      end_date: datetime) -> List[Dict]:
        """从Web Archive获取历史文章"""
        articles = []
        news_sites = [
            'www.skysports.com/football',
            'www.goal.com/en',
            'www.espn.com/soccer'
        ]
        
        for site in news_sites:
            async with self._request_semaphore:
                await asyncio.sleep(self._request_interval)
                
                params = {
                    'url': f'https://{site}',
                    'timestamp': start_date.strftime('%Y%m%d')
                }
                
                try:
                    async with self.session.get(self.WEB_ARCHIVE_BASE, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'archived_snapshots' in data:
                                snapshot_url = data['archived_snapshots']['closest']['url']
                                # 获取快照内容
                                async with self.session.get(snapshot_url) as snapshot_response:
                                    if snapshot_response.status == 200:
                                        html = await snapshot_response.text()
                                        soup = BeautifulSoup(html, 'html.parser')
                                        
                                        # 搜索包含球员和球队名称的文章
                                        for article in soup.find_all(['article', 'div'], class_=['article', 'story']):
                                            text = article.text.lower()
                                            if player.lower() in text and team.lower() in text:
                                                title = article.find(['h1', 'h2'])
                                                if title:
                                                    articles.append({
                                                        'date': start_date.isoformat(),
                                                        'title': title.text.strip(),
                                                        'source': site,
                                                        'url': snapshot_url
                                                    })
                except Exception as e:
                    print(f"Web Archive请求失败 ({site}): {str(e)}")
                    
        return articles
        
    def _get_season_dates(self, season: str) -> Dict[str, datetime]:
        """获取赛季的开始和结束日期"""
        year = int(season.split('-')[0])
        return {
            'start': datetime(year, 8, 1),  # 赛季通常8月开始
            'end': datetime(year + 1, 5, 31)  # 次年5月结束
        }
        
    def _deduplicate_articles(self, articles: List[Dict], seen_urls: Set[str]) -> List[Dict]:
        """去除重复文章"""
        unique_articles = []
        for article in articles:
            url = article['url']
            if url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        return unique_articles
        
    async def close(self):
        """关闭会话"""
        if self.session:
            await self.session.close()
            self.session = None 