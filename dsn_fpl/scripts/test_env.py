import asyncio
from pathlib import Path
from dsn_fpl.src.core.data.media_collector import MediaCollector
from dsn_fpl.src.core.data.weather_collector import WeatherCollector
from dsn_fpl.src.core.data.config import Config

async def test_media_collector():
    """测试媒体数据收集器"""
    print("\n测试媒体数据收集...")
    
    # 初始化收集器
    collector = MediaCollector(news_api_key=Config().news_api_key)
    
    # 测试连接
    await collector.initialize()
    
    # 测试数据获取
    test_cases = [
        ("Saka", "Arsenal"),
        ("Haaland", "Manchester City"),
        ("Salah", "Liverpool")
    ]
    
    for player, team in test_cases:
        print(f"\n获取 {player} ({team}) 的媒体数据:")
        articles = await collector.get_media_data(player, team, days_back=3)
        if articles:
            print(f"找到 {len(articles)} 篇相关文章")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n文章 {i}:")
                print(f"标题: {article['title']}")
                print(f"来源: {article['source']}")
                print(f"情感得分: {article['sentiment']['score']:.2f}")
                print(f"置信度: {article['sentiment']['confidence']:.2f}")
        else:
            print("未找到相关文章")
            
    # 关闭收集器
    await collector.close()

async def test_weather_collector():
    """测试天气数据收集器"""
    print("\n测试天气数据收集...")
    
    collector = WeatherCollector(api_key=Config().weather_api_key)
    
    # 测试阿森纳主场天气
    weather = collector.get_weather_data("Arsenal")
    if weather:
        print("\n阿森纳主场天气:")
        print(f"温度: {weather['temperature']}°C")
        print(f"降水量: {weather['precipitation']}mm")
        print(f"风速: {weather['wind_speed']}m/s")
        print(f"湿度: {weather['humidity']}%")
        print(f"天气状况: {weather['condition']}")

async def main():
    """主测试函数"""
    print("开始环境数据测试...")
    
    # 测试媒体数据收集
    await test_media_collector()
    
    # 测试天气数据收集
    await test_weather_collector()

if __name__ == "__main__":
    asyncio.run(main()) 