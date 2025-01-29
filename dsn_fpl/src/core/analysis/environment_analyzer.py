from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from ..data.collector import EnvironmentalData, NewsSource

@dataclass
class EnvironmentalImpact:
    """环境影响评估"""
    player_id: int
    impact_score: float  # -1 到 1
    confidence: float
    sources: List[EnvironmentalData]
    
class EnvironmentAnalyzer:
    """环境扰动分析器"""
    def __init__(self):
        self.impact_weights = {
            NewsSource.WEATHER: 0.2,
            NewsSource.NEWS: 0.15,
            NewsSource.TWITTER: 0.1,
            NewsSource.INJURY: 0.2,
            NewsSource.VENUE: 0.15,    # 场地因素权重
            NewsSource.MEDIA_SENTIMENT: 0.1,  # 媒体情感权重
            NewsSource.TEAM_FORM: 0.05,  # 球队状态权重
            NewsSource.RIVALRY: 0.05   # 对手关系权重
        }
        
    def analyze_impacts(self, 
                       env_data: List[EnvironmentalData],
                       player_ids: List[int]) -> Dict[int, EnvironmentalImpact]:
        """分析环境扰动对球员的影响"""
        impacts = {}
        
        for player_id in player_ids:
            relevant_data = self._filter_relevant_data(env_data, player_id)
            if not relevant_data:
                continue
                
            impact_score = self._compute_impact_score(relevant_data)
            confidence = self._compute_confidence(relevant_data)
            
            impacts[player_id] = EnvironmentalImpact(
                player_id=player_id,
                impact_score=impact_score,
                confidence=confidence,
                sources=relevant_data
            )
            
        return impacts
        
    def _filter_relevant_data(self, 
                            env_data: List[EnvironmentalData], 
                            player_id: int) -> List[EnvironmentalData]:
        """筛选与特定球员相关的环境数据"""
        return [
            data for data in env_data
            if not data.related_players or player_id in data.related_players
        ]
        
    def _compute_impact_score(self, data: List[EnvironmentalData]) -> float:
        """计算综合影响分数"""
        weighted_impacts = []
        
        for item in data:
            # 基础影响分数
            base_impact = item.severity * (1 if self._is_positive_event(item) else -1)
            
            # 场地因素
            if item.source == NewsSource.VENUE and item.venue_data:
                venue_multiplier = self._compute_venue_impact(item.venue_data)
                base_impact *= venue_multiplier
            
            # 天气因素
            if item.source == NewsSource.WEATHER and item.weather_details:
                weather_multiplier = self._compute_weather_impact(item.weather_details)
                base_impact *= weather_multiplier
            
            # 媒体情感
            if item.source == NewsSource.MEDIA_SENTIMENT and item.media_analysis:
                sentiment_multiplier = self._compute_media_sentiment(item.media_analysis)
                base_impact *= sentiment_multiplier
                
            # 球队状态
            if item.source == NewsSource.TEAM_FORM and item.team_stats:
                team_multiplier = self._compute_team_impact(item.team_stats)
                base_impact *= team_multiplier
                
            # 对手关系
            if item.source == NewsSource.RIVALRY and item.rivalry_info:
                rivalry_multiplier = self._compute_rivalry_impact(item.rivalry_info)
                base_impact *= rivalry_multiplier
                
            weight = self.impact_weights[item.source]
            weighted_impacts.append(base_impact * weight)
            
        return np.tanh(sum(weighted_impacts))  # 使用tanh确保结果在-1到1之间
        
    def _compute_confidence(self, data: List[EnvironmentalData]) -> float:
        """计算置信度"""
        if not data:
            return 0.0
            
        weighted_confidence = sum(
            d.confidence * self.impact_weights[d.source]
            for d in data
        )
        total_weight = sum(
            self.impact_weights[d.source]
            for d in data
        )
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
        
    def _compute_venue_impact(self, venue_data: Dict) -> float:
        """计算场地影响因子"""
        impact = 1.0
        
        # 主场优势
        if venue_data.get('is_home', False):
            impact *= 1.2
            
        # 场地状况 (0-1)
        pitch_condition = venue_data.get('pitch_condition', 1.0)
        impact *= (0.8 + pitch_condition * 0.4)  # 0.8-1.2
        
        # 观众因素 (0-1)
        crowd_factor = venue_data.get('crowd_factor', 1.0)
        impact *= (0.9 + crowd_factor * 0.2)  # 0.9-1.1
        
        return max(0.7, min(1.4, impact))
        
    def _compute_weather_impact(self, weather_details: Dict) -> float:
        """计算详细天气影响"""
        impact = 1.0
        
        # 温度影响 (适宜温度15-25度)
        temp = weather_details.get('temperature', 20)
        if temp < 5:
            impact *= 0.8
        elif temp < 15:
            impact *= 0.9
        elif temp > 30:
            impact *= 0.85
        elif temp > 25:
            impact *= 0.95
            
        # 降水影响 (0-1)
        precipitation = weather_details.get('precipitation', 0)
        impact *= (1 - precipitation * 0.3)  # 最多降低30%
        
        # 风速影响 (0-1)
        wind_speed = weather_details.get('wind_speed', 0)
        impact *= (1 - wind_speed * 0.2)  # 最多降低20%
        
        return max(0.6, min(1.2, impact))
        
    def _compute_media_sentiment(self, media_analysis: Dict) -> float:
        """计算媒体情感影响"""
        # 情感分数 (-1到1)
        sentiment_score = media_analysis.get('sentiment_score', 0)
        # 报道强度 (0-1)
        coverage_intensity = media_analysis.get('coverage_intensity', 0.5)
        
        # 计算影响
        sentiment_impact = 1 + (sentiment_score * coverage_intensity * 0.3)
        return max(0.7, min(1.3, sentiment_impact))
        
    def _compute_team_impact(self, team_stats: Dict) -> float:
        """计算球队状态影响"""
        impact = 1.0
        
        # 球队状态 (0-1)
        form = team_stats.get('form', 0.5)
        impact *= (0.8 + form * 0.4)  # 0.8-1.2
        
        # 进球/失球差
        goals_scored = team_stats.get('goals_scored', 0)
        goals_conceded = team_stats.get('goals_conceded', 0)
        if goals_scored > goals_conceded:
            impact *= 1.1
        elif goals_scored < goals_conceded:
            impact *= 0.9
            
        return max(0.7, min(1.3, impact))
        
    def _compute_rivalry_impact(self, rivalry_info: Dict) -> float:
        """计算对手关系影响"""
        impact = 1.0
        
        # 历史战绩
        wins = rivalry_info.get('historical_wins', 0)
        losses = rivalry_info.get('historical_losses', 0)
        if wins > losses:
            impact *= 1.1
        elif wins < losses:
            impact *= 0.95
            
        # 比赛重要性 (0-1)
        importance = rivalry_info.get('importance', 0.5)
        impact *= (0.9 + importance * 0.2)  # 0.9-1.1
        
        return max(0.8, min(1.2, impact))
        
    def _is_positive_event(self, data: EnvironmentalData) -> bool:
        """判断是否为正面事件"""
        if data.source == NewsSource.WEATHER:
            return data.weather_details.get('temperature', 20) >= 15 and \
                   data.weather_details.get('precipitation', 0) < 0.3 and \
                   data.weather_details.get('wind_speed', 0) < 0.3
        elif data.source == NewsSource.VENUE:
            return data.venue_data.get('is_home', False) and \
                   data.venue_data.get('pitch_condition', 0.5) > 0.7
        elif data.source == NewsSource.MEDIA_SENTIMENT:
            return data.media_analysis.get('sentiment_score', 0) > 0
        elif data.source == NewsSource.TEAM_FORM:
            return data.team_stats.get('form', 0.5) > 0.6
        elif data.source == NewsSource.RIVALRY:
            wins = data.rivalry_info.get('historical_wins', 0)
            losses = data.rivalry_info.get('historical_losses', 0)
            return wins >= losses
        elif data.source in [NewsSource.NEWS, NewsSource.TWITTER]:
            return float(data.description.split(':')[0]) > 0
        return True 