from typing import Dict, List, Optional
import requests
import feedparser
from datetime import datetime, timedelta
from textblob import TextBlob
from bs4 import BeautifulSoup
from ..data.collector import EnvironmentalData, NewsSource
from ..cache.cache_manager import CacheManager
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
        
    def get_media_data(self, player: str, team: str, days_back: int = 7) -> List[Dict]:
        """获取球员相关的媒体数据
        
        Args:
            player: 球员名称
            team: 球队名称
            days_back: 获取过去几天的数据
            
        Returns:
            包含媒体数据的列表
        """
        cache_key = f"media_{player}_{team}_{days_back}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
            
        articles = []
        
        # 从RSS源获取数据
        rss_articles = self._get_rss_articles(player, team, days_back)
        if rss_articles:
            articles.extend(rss_articles)
            
        # 如果配置了NewsAPI，也从那里获取数据
        if self.news_api_key:
            news_articles = self._get_news_articles(player, team, days_back)
            if news_articles:
                articles.extend(news_articles)
                
        # 去重（基于标题）
        unique_articles = self._deduplicate_articles(articles)
        
        # 情感分析
        analyzed_articles = self.sentiment_analyzer.analyze_articles(unique_articles)
        
        # 缓存结果
        self.cache.set(cache_key, analyzed_articles)
        
        return analyzed_articles
        
    def create_media_impact(self, player: str, team: str) -> Optional[EnvironmentalData]:
        """创建媒体环境影响数据"""
        articles = self.get_media_data(player, team, days_back=3)  # 只看最近3天的新闻
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
            severity=abs(avg_score),  # 情感强度作为严重程度
            confidence=min(total_confidence / len(articles), 0.9),  # 平均置信度，最高0.9
            event_type=self._determine_media_type(avg_score),
            media_analysis={  # 使用media_analysis字段
                'article_count': len(articles),
                'sentiment_score': avg_score,
                'recent_headlines': recent_headlines,
                'coverage_intensity': min(len(articles) / 10, 1.0)  # 添加覆盖强度
            }
        )
        
    def _get_rss_articles(self, player: str, team: str, days_back: int) -> List[Dict]:
        """从RSS源获取文章"""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for source, url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    # 检查发布日期
                    pub_date = datetime(*entry.published_parsed[:6])
                    if pub_date < cutoff_date:
                        continue
                        
                    # 检查是否相关
                    title = entry.title.lower()
                    description = entry.get('description', '').lower()
                    if player.lower() in title + description or team.lower() in title + description:
                        # 清理HTML内容
                        clean_desc = BeautifulSoup(description, 'html.parser').get_text()
                        
                        articles.append({
                            'title': entry.title,
                            'description': clean_desc,
                            'source': source,
                            'url': entry.link,
                            'published_at': pub_date.isoformat()
                        })
            except Exception as e:
                print(f"获取RSS源 {source} 数据时出错: {str(e)}")
                
        return articles
        
    def _get_news_articles(self, player: str, team: str, days_back: int) -> List[Dict]:
        """从NewsAPI获取文章"""
        if not self.news_api_key:
            return []
            
        try:
            from newsapi import NewsApiClient
            newsapi = NewsApiClient(api_key=self.news_api_key)
            
            # 构建查询
            query = f"{player} {team}"
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            response = newsapi.get_everything(
                q=query,
                from_param=from_date,
                language='en',
                sort_by='relevancy'
            )
            
            articles = []
            for article in response.get('articles', []):
                # 清理HTML内容
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
        """确定媒体影响类型"""
        if sentiment_score > 0.3:
            return 'Very_Positive'
        elif sentiment_score > 0:
            return 'Positive'
        elif sentiment_score > -0.3:
            return 'Negative'
        else:
            return 'Very_Negative' 