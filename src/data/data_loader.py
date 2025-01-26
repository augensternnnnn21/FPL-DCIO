import os
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

class FPLDataLoader:
    """FPL数据加载器"""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.merged_data_path = self.base_path / 'cleaned_merged_seasons.csv'
        self.master_team_path = self.base_path / 'master_team_list.csv'
    
    def load_data(self, season: Optional[str] = None) -> pd.DataFrame:
        """加载数据
        Args:
            season: 可选的赛季，如果不提供则加载合并数据
        """
        if season:
            season_path = self.base_path / season / 'gws'
            return self._load_season_data(season_path)
        else:
            return self._load_merged_data()
    
    def _load_merged_data(self) -> pd.DataFrame:
        """加载合并的赛季数据"""
        if not self.merged_data_path.exists():
            raise FileNotFoundError(f"找不到合并数据文件: {self.merged_data_path}")
        return pd.read_csv(self.merged_data_path)
    
    def _load_season_data(self, season_path: Path) -> pd.DataFrame:
        """加载单个赛季的数据"""
        if not season_path.exists():
            raise FileNotFoundError(f"找不到赛季目录: {season_path}")
        
        all_gw_data = []
        for gw in range(1, 39):
            gw_file = season_path / f'gw{gw}.csv'
            if gw_file.exists():
                try:
                    gw_data = pd.read_csv(gw_file, encoding='latin1')
                    gw_data['gameweek'] = gw
                    all_gw_data.append(gw_data)
                except Exception as e:
                    print(f"处理比赛周{gw}时出错: {e}")
        
        if not all_gw_data:
            raise ValueError(f"无法加载任何比赛周数据: {season_path}")
        
        return pd.concat(all_gw_data, ignore_index=True)
    
    def load_team_list(self) -> pd.DataFrame:
        """加载主球队列表"""
        if not self.master_team_path.exists():
            raise FileNotFoundError(f"找不到球队列表文件: {self.master_team_path}")
        return pd.read_csv(self.master_team_path)