import asyncio
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import pytz
import sys
import os

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from dsn_fpl.src.core.data.news_collector import NewsCollector

async def collect_season_news(season: str = "2023-24"):
    """收集指定赛季的新闻数据
    
    Args:
        season: 赛季标识
    """
    # 初始化收集器
    collector = NewsCollector(Path(project_root) / "data")
    
    # 读取赛季数据
    fixtures_path = Path(project_root) / "data" / season / "fixtures.csv"
    teams_path = Path(project_root) / "data" / season / "teams.csv"
    
    if not fixtures_path.exists():
        logging.error(f"找不到赛季数据文件: {fixtures_path}")
        return
        
    if not teams_path.exists():
        logging.error(f"找不到球队数据文件: {teams_path}")
        return
        
    # 读取数据
    fixtures_df = pd.read_csv(fixtures_path)
    teams_df = pd.read_csv(teams_path)
    
    # 确保时间列是datetime类型
    fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time'])
    
    # 按时间排序
    fixtures_df = fixtures_df.sort_values('kickoff_time')
    
    # 获取所有球队名称映射
    team_name_map = dict(zip(teams_df['team_id'], teams_df['name']))
    
    # 收集每场比赛的新闻
    for _, match in fixtures_df.iterrows():
        try:
            # 获取球队名称
            home_team = team_name_map[match['team_h']]
            away_team = team_name_map[match['team_a']]
            match_time = match['kickoff_time'].to_pydatetime().replace(tzinfo=pytz.UTC)
            
            # 收集主队新闻
            logging.info(f"收集 {home_team} vs {away_team} 的新闻...")
            home_news = await collector.collect_match_news(
                home_team, match_time, away_team
            )
            collector.save_news(season, home_news)
            
            # 收集客队新闻
            away_news = await collector.collect_match_news(
                away_team, match_time, home_team
            )
            collector.save_news(season, away_news)
            
            # 随机延时，避免请求过于频繁
            await asyncio.sleep(5)
            
        except Exception as e:
            logging.error(f"处理比赛 {home_team} vs {away_team} 时出错: {str(e)}")
            continue
            
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('news_collection.log')
        ]
    )
    
    # 运行收集器
    asyncio.run(collect_season_news()) 