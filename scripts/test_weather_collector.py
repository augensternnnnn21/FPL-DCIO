import asyncio
import sys
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dsn_fpl.src.core.data.historical_weather_collector import HistoricalWeatherCollector

async def main():
    """测试历史天气数据收集"""
    print("开始测试历史天气数据收集...")
    start_time = time.time()
    
    # 初始化收集器
    collector = HistoricalWeatherCollector(
        data_path=project_root / "data"
    )
    
    try:
        # 测试最近的赛季
        test_season = "2023-24"
        print(f"\n测试 {test_season} 赛季的天气数据收集...")
        
        # 收集数据
        await collector.collect_historical_weather(test_season)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        print(f"\n测试完成！")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"平均每场比赛耗时: {elapsed_time/380:.2f} 秒")  # 38轮 * 10场/轮
        
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
    finally:
        await collector.close()

if __name__ == "__main__":
    asyncio.run(main()) 