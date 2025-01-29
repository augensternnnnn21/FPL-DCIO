from datetime import datetime
from dsn_fpl.src.core.config import Config
from dsn_fpl.src.core.data.weather_collector import WeatherCollector
from dsn_fpl.src.core.data.media_collector import MediaCollector

def test_weather_api():
    """测试天气API"""
    config = Config()
    if not config.has_weather_api:
        print("❌ 未配置天气API密钥")
        return

    collector = WeatherCollector(api_key=config.weather_api_key)
    try:
        # 测试阿森纳主场的天气数据
        weather_data = collector.get_weather_data('Arsenal')
        if weather_data:
            print("✅ 天气API测试成功")
            print(f"温度: {weather_data['temperature']}°C")
            print(f"降水: {weather_data['precipitation']}mm")
            print(f"风速: {weather_data['wind_speed']:.1f}m/s")
            print(f"湿度: {weather_data['humidity']}%")
            print(f"天气状况: {weather_data['condition']}")
        else:
            print("❌ 获取天气数据失败")
    except Exception as e:
        print(f"❌ 天气API测试失败: {e}")

def test_news_api():
    """测试新闻API和情感分析"""
    config = Config()
    if not config.has_news_api:
        print("❌ 未配置新闻API密钥")
        return

    collector = MediaCollector(news_api_key=config.news_api_key)
    try:
        # 测试获取有关阿森纳萨卡的新闻
        articles = collector.get_media_data("Saka", "Arsenal", days_back=3)
        if articles:
            print("✅ 新闻API测试成功")
            print(f"获取到 {len(articles)} 篇相关文章")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n文章 {i}:")
                print(f"标题: {article['title']}")
                print(f"来源: {article['source']}")
                print(f"情感得分: {article['sentiment']['score']:.2f}")
                print(f"主观性: {article['sentiment']['subjectivity']:.2f}")
                print(f"置信度: {article['sentiment']['confidence']:.2f}")
                
            # 测试媒体影响分析
            impact = collector.create_media_impact("Saka", "Arsenal")
            if impact:
                print("\n媒体影响分析:")
                print(f"描述: {impact.description}")
                print(f"严重程度: {impact.severity:.2f}")
                print(f"置信度: {impact.confidence:.2f}")
                print(f"事件类型: {impact.event_type}")
        else:
            print("❌ 未找到相关新闻")
    except Exception as e:
        print(f"❌ 新闻API测试失败: {e}")

def main():
    """主函数"""
    print("开始测试环境配置...")
    print("\n1. 测试天气API")
    print("-" * 50)
    test_weather_api()
    
    print("\n2. 测试新闻API和情感分析")
    print("-" * 50)
    test_news_api()

if __name__ == "__main__":
    main() 