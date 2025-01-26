import os
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

class FPLDataLoader:
    """FPL数据加载器"""
    def __init__(self, seasons_path: Path, teams_path: Path):
        self.seasons_path = Path(seasons_path)
        self.teams_path = Path(teams_path)
        
        print(f"🔧 初始化数据加载器")
        print(f"📂 赛季数据路径: {self.seasons_path}")
        print(f"📂 球队数据路径: {self.teams_path}")
        
        # 验证文件存在性
        if not self.seasons_path.exists():
            raise FileNotFoundError(f"赛季数据文件不存在: {self.seasons_path}")
        if not self.teams_path.exists():
            raise FileNotFoundError(f"球队数据文件不存在: {self.teams_path}")
    
    def load_data(self) -> pd.DataFrame:
        """直接加载赛季数据"""
        try:
            df = pd.read_csv(self.seasons_path)
            print(f"✅ 成功加载赛季数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"❌ 加载赛季数据失败: {str(e)}")
            raise
    
    def load_team_list(self) -> pd.DataFrame:
        """加载球队数据"""
        try:
            df = pd.read_csv(self.teams_path)
            print(f"✅ 成功加载球队数据，共 {len(df)} 支球队")
            return df
        except Exception as e:
            print(f"❌ 加载球队数据失败: {str(e)}")
            raise

    