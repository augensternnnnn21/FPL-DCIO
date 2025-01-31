import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dsn_fpl.src.core.data.historical_weather_collector import HistoricalWeatherCollector

async def main():
    """收集历史天气数据的主函数"""
    # 初始化收集器
    collector = HistoricalWeatherCollector(
        data_path=project_root / "data"
    )
    
    try:
        # 获取所有赛季
        seasons = [
            "2016-17", "2017-18", "2018-19", "2019-20",
            "2020-21", "2021-22", "2022-23", "2023-24"
        ]
        
        # 收集每个赛季的数据
        for season in seasons:
            print(f"\n开始收集 {season} 赛季的天气数据...")
            await collector.collect_historical_weather(season)
            print(f"完成 {season} 赛季的数据收集")
            
    except Exception as e:
        print(f"收集数据时出错: {str(e)}")
    finally:
        await collector.close()

if __name__ == "__main__":
    asyncio.run(main()) 