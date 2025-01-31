from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from textblob import TextBlob
from bs4 import BeautifulSoup
from ..data.collector import EnvironmentalData, NewsSource
from .cache.cache_manager import CacheManager
from .sentiment_analyzer import SentimentAnalyzer

class MediaCollector:
    """媒体数据收集器"""
    
    RSS_FEEDS = {
        'bbc-sport': 'http://feeds.bbci.co.uk/sport/football/rss.xml',
        'sky-sports': 'https://www.skysports.com/rss/football',
        'guardian': 'https://www.theguardian.com/football/rss',
        'goal': 'https://www.goal.com/feeds/en/news'
    }
    
    def __init__(self, news_api_key: Optional[str] = None, cache_ttl: int = 3600):
        """初始化收集器
        
        Args:
            news_api_key: NewsAPI的API密钥（可选）
            cache_ttl: 缓存过期时间（秒）
        """
        self.news_api_key = news_api_key
        self.cache = CacheManager(cache_dir=".cache/media", ttl=cache_ttl)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.session = None
        self._request_semaphore = asyncio.Semaphore(1)  # 限制为单个请求
        self._last_request_time = datetime.now()
        self._request_interval = 2.0  # 增加请求间隔到2秒
        self.available_feeds = {}  # 存储可用的RSS源
        
        # NewsAPI请求限制
        self._news_api_requests = 0
        self._news_api_reset_time = datetime.now()
        self._news_api_max_requests = 45  # 设置为45，留一些余量
        
    async def initialize(self):
        """初始化并测试连接"""
        await self._ensure_session()
        await self._test_feeds()
        if not self.available_feeds:
            print("警告: 没有可用的RSS源")
            
    async def _test_feeds(self):
        """测试所有RSS源的可用性"""
        print("测试RSS源连接...")
        for source, url in self.RSS_FEEDS.items():
            try:
                async with self.session.get(url, ssl=False, timeout=5) as response:
                    if response.status == 200:
                        self.available_feeds[source] = url
                        print(f"RSS源 {source} 连接成功")
                    else:
                        print(f"RSS源 {source} 返回状态码 {response.status}")
            except Exception as e:
                print(f"RSS源 {source} 连接失败: {str(e)}")
                
    async def _ensure_session(self):
        """确保aiohttp会话已创建"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
    async def _wait_for_rate_limit(self):
        """等待请求间隔"""
        now = datetime.now()
        elapsed = (now - self._last_request_time).total_seconds()
        if elapsed < self._request_interval:
            await asyncio.sleep(self._request_interval - elapsed)
        self._last_request_time = datetime.now()
            
    async def _check_news_api_rate_limit(self) -> bool:
        """检查NewsAPI速率限制"""
        now = datetime.now()
        # 如果过了12小时，重置计数器
        if (now - self._news_api_reset_time).total_seconds() > 12 * 3600:
            self._news_api_requests = 0
            self._news_api_reset_time = now
            
        # 检查是否超过限制
        if self._news_api_requests >= self._news_api_max_requests:
            hours_until_reset = 12 - (now - self._news_api_reset_time).total_seconds() / 3600
            print(f"NewsAPI达到速率限制，还需等待 {hours_until_reset:.1f} 小时")
            return False
            
        return True
        
    async def get_media_data(self, player_name: str, team_name: str, days_back: int = 7, use_cache_only: bool = False) -> List[Dict]:
        """异步获取媒体数据
        
        Args:
            player_name: 球员名称
            team_name: 球队名称
            days_back: 获取过去几天的数据
            use_cache_only: 是否只使用缓存数据
        """
        # 检查缓存
        cache_key = f"media_{player_name}_{team_name}_{days_back}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            print(f"使用缓存的媒体数据: {len(cached_data)} 篇文章")
            return cached_data
            
        # 如果只使用缓存且没有缓存数据，返回空列表
        if use_cache_only:
            return []
            
        # 确保已初始化
        if not hasattr(self, 'available_feeds'):
            await self.initialize()
            
        articles = []
        
        # 限制并发请求数
        async with self._request_semaphore:
            # RSS源数据获取
            for source, url in self.available_feeds.items():
                await self._wait_for_rate_limit()
                try:
                    source_articles = await self._get_rss_articles_with_retry(source, url, f"{player_name} {team_name}")
                    if source_articles:
                        articles.extend(source_articles)
                        print(f"从 {source} 获取到 {len(source_articles)} 篇文章")
                except Exception as e:
                    print(f"获取RSS源 {source} 数据时出错: {str(e)}")
                    continue
            
            # 如果RSS源没有数据，尝试使用NewsAPI
            if not articles and self.news_api_key:
                if await self._check_news_api_rate_limit():
                    try:
                        news_articles = await self._get_news_articles(f"{player_name} {team_name}", days_back)
                        if news_articles:
                            articles.extend(news_articles)
                            print(f"从 NewsAPI 获取到 {len(news_articles)} 篇文章")
                            self._news_api_requests += 1
                    except Exception as e:
                        print(f"获取NewsAPI数据时出错: {str(e)}")
                else:
                    print("跳过NewsAPI请求，使用缓存数据")
        
        # 去重
        unique_articles = self._deduplicate_articles(articles)
        
        # 分析情感
        for article in unique_articles:
            sentiment = self.sentiment_analyzer.analyze_text(
                f"{article['title']} {article.get('description', '')}"
            )
            article['sentiment'] = sentiment
            
        # 缓存结果
        if unique_articles:
            self.cache.set(cache_key, unique_articles)
            print(f"为 {player_name} 缓存了 {len(unique_articles)} 篇文章")
        
        return unique_articles
        
    async def _get_rss_articles_with_retry(self, source: str, url: str, term: str, max_retries: int = 3) -> List[Dict]:
        """带重试机制的RSS文章获取"""
        for attempt in range(max_retries):
            try:
                async with self.session.get(url, ssl=False) as response:
                    if response.status != 200:
                        raise aiohttp.ClientError(f"HTTP {response.status}")
                        
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    articles = []
                    cutoff_date = datetime.now() - timedelta(days=7)
                    
                    for entry in feed.entries:
                        try:
                            # 检查发布日期
                            pub_date = datetime(*entry.published_parsed[:6])
                            if pub_date < cutoff_date:
                                continue
                                
                            # 检查是否相关
                            title = entry.title.lower()
                            description = entry.get('description', '').lower()
                            if term.lower() in title + description:
                                # 清理HTML内容
                                clean_desc = BeautifulSoup(description, 'html.parser').get_text()
                                
                                articles.append({
                                    'title': entry.title,
                                    'description': clean_desc,
                                    'source': source,
                                    'url': entry.link,
                                    'published_at': pub_date.isoformat()
                                })
                        except Exception:
                            continue
                            
                    return articles
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(1 * (attempt + 1))  # 指数退避
                
        return []
        
    async def _get_news_articles(self, term: str, days_back: int) -> List[Dict]:
        """异步从NewsAPI获取文章"""
        if not self.news_api_key:
            return []
            
        try:
            from newsapi.newsapi_client import NewsApiClient
            newsapi = NewsApiClient(api_key=self.news_api_key)
            
            # 构建查询
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            response = await asyncio.to_thread(
                newsapi.get_everything,
                q=term,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=5  # 限制每次请求的文章数量
            )
            
            if response.get('status') == 'error':
                print(f"NewsAPI错误: {response.get('message')}")
                return []
            
            articles = []
            for article in response.get('articles', []):
                description = BeautifulSoup(article.get('description', ''), 'html.parser').get_text()
                
                articles.append({
                    'title': article['title'],
                    'description': description,
                    'source': article['source']['name'],
                    'url': article['url'],
                    'published_at': article['publishedAt']
                })
                
            return articles
        except Exception as e:
            print(f"获取NewsAPI数据时出错: {str(e)}")
            return []
            
    async def create_media_impact(self, player: str, team: str) -> Optional[EnvironmentalData]:
        """异步创建媒体环境影响数据"""
        articles = await self.get_media_data(player, team, days_back=3)
        if not articles:
            return None
            
        # 计算整体情感得分
        total_score = 0
        total_confidence = 0
        for article in articles:
            sentiment = article['sentiment']
            total_score += sentiment['score'] * sentiment['confidence']
            total_confidence += sentiment['confidence']
            
        if total_confidence > 0:
            avg_score = total_score / total_confidence
        else:
            avg_score = 0
            
        # 生成描述
        description = self._generate_media_description(articles)
        
        # 获取最近的头条
        recent_headlines = [a['title'] for a in articles[:3]]
        
        return EnvironmentalData(
            source=NewsSource.NEWS,
            timestamp=datetime.now(),
            description=description,
            severity=abs(avg_score),
            confidence=min(total_confidence / len(articles), 0.9),
            event_type=self._determine_media_type(avg_score),
            media_analysis={
                'article_count': len(articles),
                'sentiment_score': avg_score,
                'recent_headlines': recent_headlines,
                'coverage_intensity': min(len(articles) / 10, 1.0)
            }
        )
        
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """去除重复文章"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title = article['title'].lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
                
        return unique_articles
        
    def _generate_media_description(self, articles: List[Dict]) -> str:
        """生成媒体影响描述"""
        total_sentiment = sum(a['sentiment']['score'] for a in articles)
        avg_sentiment = total_sentiment / len(articles) if articles else 0
        
        if avg_sentiment > 0.3:
            sentiment_desc = "非常正面"
        elif avg_sentiment > 0:
            sentiment_desc = "偏正面"
        elif avg_sentiment > -0.3:
            sentiment_desc = "偏负面"
        else:
            sentiment_desc = "非常负面"
            
        return f"最近新闻报道整体{sentiment_desc}，共{len(articles)}篇相关报道"
        
    def _determine_media_type(self, sentiment_score: float) -> str:
        """确定媒体类型"""
        if sentiment_score > 0.3:
            return 'Positive_News'
        elif sentiment_score < -0.3:
            return 'Negative_News'
        return 'Neutral_News'
        
    async def close(self):
        """关闭会话"""
        if self.session:
            await self.session.close()
            self.session = None 