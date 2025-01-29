import os
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

class FPLDataLoader:
    """FPL数据加载器"""
    def __init__(self, seasons_path: str, teams_path: str):
        self.seasons_path = seasons_path
        self.teams_path = teams_path
        
        print(f"🔧 初始化数据加载器")
        print(f"📂 赛季数据路径: {self.seasons_path}")
        print(f"📂 球队数据路径: {self.teams_path}")
        
        # 验证文件存在性
        if not os.path.exists(self.seasons_path):
            raise FileNotFoundError(f"赛季数据文件不存在: {self.seasons_path}")
        if not os.path.exists(self.teams_path):
            raise FileNotFoundError(f"球队数据文件不存在: {self.teams_path}")
    
    def load_data(self, season: str = None) -> pd.DataFrame:
        """加载数据"""
        try:
            # 加载数据
            df = pd.read_csv(self.seasons_path, low_memory=False)
            print(f"✅ 成功加载赛季数据，共 {len(df)} 条记录")
            
            # 如果指定了赛季，筛选数据
            if season:
                # 打印所有可用的赛季
                available_seasons = df['season_x'].unique()
                print(f"可用赛季: {available_seasons}")
                
                # 尝试精确匹配
                season_data = df[df['season_x'] == season]
                
                # 如果没有找到，尝试模糊匹配
                if len(season_data) == 0:
                    season_data = df[df['season_x'].str.contains(season, na=False)]
                
                print(f"✅ 筛选 {season} 赛季数据，共 {len(season_data)} 条记录")
                
                # 如果仍然没有找到数据，使用最近的赛季
                if len(season_data) == 0:
                    print(f"⚠️ 警告: 未找到 {season} 赛季的数据")
                    # 按年份排序赛季
                    sorted_seasons = sorted(available_seasons, 
                                         key=lambda x: int(x.split('-')[0]))
                    # 找到最接近的赛季
                    target_year = int(season.split('-')[0])
                    closest_season = min(sorted_seasons, 
                                       key=lambda x: abs(int(x.split('-')[0]) - target_year))
                    print(f"使用最接近的赛季 {closest_season} 作为替代")
                    season_data = df[df['season_x'] == closest_season]
                
                if len(season_data) == 0:
                    raise ValueError(f"无法找到有效的赛季数据")
                    
                return season_data
            
            return df
            
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            raise
    
    def load_team_list(self) -> pd.DataFrame:
        """加载球队列表"""
        try:
            team_data = pd.read_csv(self.teams_path)
            print(f"✅ 成功加载球队数据，共 {len(team_data)} 支球队")
            return team_data
        except Exception as e:
            print(f"❌ 球队数据加载失败: {str(e)}")
            raise

    