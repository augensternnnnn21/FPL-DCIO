import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class FPLPreprocessor:
    """FPL数据预处理器"""
    def __init__(self):
        self.position_mapping = {
            'GKP': 'GK',
            'GK': 'GK',
            'DEF': 'DEF',
            'MID': 'MID',
            'FWD': 'FWD',
            'FOR': 'FWD'
        }
    
    def preprocess(self, data: pd.DataFrame, team_list: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """预处理数据"""
        data = data.copy()
        
        # 1. 处理比赛周
        data = self._process_gameweek(data)
        
        # 2. 处理球队信息
        data = self._process_team(data, team_list)
        
        # 3. 处理位置信息
        data = self._process_position(data)
        
        # 4. 处理数值特征
        data = self._process_features(data)
        
        return data
    
    def _process_gameweek(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理比赛周信息"""
        if 'gw' not in data.columns:
            if 'round' in data.columns:
                data['gw'] = data['round']
            elif 'gameweek' in data.columns:
                data['gw'] = data['gameweek']
        return data
    
    def _process_team(self, data: pd.DataFrame, team_list: Optional[pd.DataFrame]) -> pd.DataFrame:
        """处理球队信息"""
        if team_list is not None:
            # 使用主球队列表进行映射
            team_mapping = dict(zip(team_list['raw_name'], team_list['standard_name']))
            if 'team_x' in data.columns:
                data['team'] = data['team_x'].map(team_mapping)
            elif 'team' in data.columns:
                data['team'] = data['team'].map(team_mapping)
        else:
            # 直接使用现有的球队信息
            if 'team_x' in data.columns:
                data['team'] = data['team_x']
        
        if 'team' not in data.columns:
            raise ValueError("无法找到或处理球队信息")
        
        return data
    
    def _process_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理位置信息"""
        data['position'] = data['position'].map(self.position_mapping)
        if data['position'].isna().any():
            raise ValueError("存在无效的位置信息")
        return data
    
    def _process_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理数值特征"""
        # 基本特征
        numeric_cols = ['value', 'total_points', 'minutes', 'goals_scored', 
                       'assists', 'clean_sheets', 'goals_conceded']
        
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 填充缺失值
        data = data.fillna(0)
        
        return data 