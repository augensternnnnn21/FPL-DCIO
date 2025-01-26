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
        """加载数据
        
        Args:
            season: 可选的赛季筛选条件
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            # 加载数据
            df = pd.read_csv(self.seasons_path)
            print(f"✅ 成功加载赛季数据，共 {len(df)} 条记录")
            
            # 如果指定了赛季，筛选数据
            if season:
                df = df[df['season_x'] == season]
                print(f"✅ 筛选 {season} 赛季数据，共 {len(df)} 条记录")
            
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

    