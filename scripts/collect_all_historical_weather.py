import asyncio
import sys
from pathlib import Path
import time
import json
from datetime import datetime
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from dsn_fpl.src.core.data.historical_weather_collector import HistoricalWeatherCollector

class CollectionProgress:
    def __init__(self):
        self.progress_file = Path(".cache/weather_collection_progress.json")
        self.load_progress()
        
    def load_progress(self):
        """加载进度数据"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    self.progress = json.load(f)
                    logging.info(f"加载现有进度: {self.progress}")
            except:
                logging.warning("加载进度文件失败，初始化新进度")
                self.init_progress()
        else:
            logging.info("进度文件不存在，初始化新进度")
            self.init_progress()
            
    def init_progress(self):
        """初始化进度数据"""
        self.progress = {
            'completed_seasons': [],
            'last_update': None,
            'total_matches': 0,
            'start_time': None
        }
        
    def save_progress(self):
        """保存进度数据"""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
        logging.info(f"保存进度: {self.progress}")
            
    def update_progress(self, season: str, matches: int):
        """更新进度"""
        self.progress['completed_seasons'].append(season)
        self.progress['total_matches'] += matches
        self.progress['last_update'] = datetime.now().isoformat()
        self.save_progress()
        logging.info(f"更新进度: 赛季 {season} 完成，总计 {self.progress['total_matches']} 场比赛")
        
    def is_season_completed(self, season: str) -> bool:
        """检查赛季是否已完成"""
        return season in self.progress['completed_seasons']
        
    def start_collection(self):
        """开始收集"""
        if not self.progress['start_time']:
            self.progress['start_time'] = datetime.now().isoformat()
            self.save_progress()
            logging.info("开始新的数据收集任务")

async def main():
    """收集所有历史天气数据的主函数"""
    logging.info("开始收集所有历史天气数据...")
    
    # 确保目录存在
    weather_dir = project_root / "data" / "historical" / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"确保目录存在: {weather_dir}")
    
    # 初始化进度管理器
    progress = CollectionProgress()
    progress.start_collection()
    
    # 初始化收集器
    collector = HistoricalWeatherCollector(
        data_path=project_root / "data"
    )
    
    try:
        # 所有赛季
        seasons = [
            "2016-17", "2017-18", "2018-19", "2019-20",
            "2020-21", "2021-22", "2022-23", "2023-24"
        ]
        
        total_seasons = len(seasons)
        completed_seasons = len(progress.progress['completed_seasons'])
        
        logging.info(f"总进度: {completed_seasons}/{total_seasons} 赛季完成")
        logging.info(f"已收集 {progress.progress['total_matches']} 场比赛的天气数据")
        
        if progress.progress['start_time']:
            start_time = datetime.fromisoformat(progress.progress['start_time'])
            elapsed = datetime.now() - start_time
            logging.info(f"已运行时间: {elapsed}")
        
        # 收集每个赛季的数据
        for season in seasons:
            if progress.is_season_completed(season):
                logging.info(f"赛季 {season} 已完成，跳过")
                continue
                
            logging.info(f"开始收集 {season} 赛季的天气数据...")
            season_start_time = time.time()
            
            # 收集数据
            await collector.collect_historical_weather(season)
            
            # 更新进度
            season_time = time.time() - season_start_time
            logging.info(f"完成 {season} 赛季的数据收集")
            logging.info(f"赛季耗时: {season_time:.2f} 秒")
            
            # 检查是否成功保存了数据
            season_files = list(weather_dir.glob(f"{season}_*_weather.csv"))
            logging.info(f"本赛季生成的文件数: {len(season_files)}")
            for file in season_files:
                logging.info(f"生成文件: {file.name} ({file.stat().st_size} bytes)")
            
            # 估算剩余时间
            remaining_seasons = total_seasons - (completed_seasons + 1)
            if remaining_seasons > 0:
                estimated_remaining = season_time * remaining_seasons
                logging.info(f"预估剩余时间: {estimated_remaining/60:.1f} 分钟")
            
            progress.update_progress(season, 380)  # 每个赛季380场比赛
            
        logging.info("所有数据收集完成！")
        logging.info(f"总计收集 {progress.progress['total_matches']} 场比赛的天气数据")
        
        if progress.progress['start_time']:
            total_time = datetime.now() - datetime.fromisoformat(progress.progress['start_time'])
            logging.info(f"总耗时: {total_time}")
            
    except Exception as e:
        logging.error(f"收集数据时出错: {str(e)}", exc_info=True)
    finally:
        await collector.close()

if __name__ == "__main__":
    asyncio.run(main()) 