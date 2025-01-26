import os
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

class FPLDataLoader:
    """FPL数据加载器"""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.merged_data_path = self.base_path / 'cleaned_merged_seasons.csv'
        self.team_list_path = self.base_path / 'master_team_list.csv'
    
    def load_merged_data(self) -> pd.DataFrame:
        """加载合并的赛季数据"""
        if not self.merged_data_path.exists():
            raise FileNotFoundError(f"找不到合并数据文件: {self.merged_data_path}")
        return pd.read_csv(self.merged_data_path)
    
    def load_team_list(self) -> pd.DataFrame:
        """加载球队列表"""
        if not self.team_list_path.exists():
            raise FileNotFoundError(f"找不到球队列表文件: {self.team_list_path}")
        return pd.read_csv(self.team_list_path)
    
    def load_season_data(self, season: str) -> pd.DataFrame:
        """加载单个赛季的数据"""
        season_path = self.base_path / season / f"{season}_processed.csv"
        if not season_path.exists():
            raise FileNotFoundError(f"找不到赛季数据文件: {season_path}")
        return pd.read_csv(season_path) 