from typing import List, Dict

class NewsCollector:
    """新闻数据采集器"""
    def __init__(self):
        self.sources = [
            'footballapi-api.com',  # 专业足球新闻API
            'newsapi.org',          # 通用新闻API
            'premierleague.com/rss' # 英超官方RSS
        ]
        
    async def get_team_news(self, team: str) -> List[Dict]:
        """获取球队新闻"""
        news = []
        for source in self.sources:
            try:
                if source == 'premierleague.com/rss':
                    news.extend(await self._scrape_pl_news(team))
                else:
                    news.extend(await self._fetch_api_news(source, team))
            except:
                continue
        return news 