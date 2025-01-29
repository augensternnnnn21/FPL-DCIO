from typing import Dict, Optional
from datetime import datetime
from ..data.collector import EnvironmentalData, NewsSource

class VenueCollector:
    """场地数据收集器"""
    def __init__(self):
        # 球场基础数据
        self.stadium_data = {
            'Arsenal': {
                'name': 'Emirates Stadium',
                'capacity': 60704,
                'pitch_quality': 0.95,  # 草坪质量评分 (0-1)
                'roof_coverage': 1.0,    # 看台遮盖率 (0-1)
                'pitch_size': {'length': 105, 'width': 68},  # 标准尺寸
                'surface_type': 'Desso GrassMaster',  # 混合草坪
                'drainage_rating': 0.95,  # 排水系统评分
                'floodlight_lux': 2000,  # 照明亮度
            },
            'Aston Villa': {
                'name': 'Villa Park',
                'capacity': 42785,
                'pitch_quality': 0.9,
                'roof_coverage': 0.8,
                'pitch_size': {'length': 105, 'width': 68},
                'surface_type': 'Desso GrassMaster',
                'drainage_rating': 0.85,
                'floodlight_lux': 1800,
            },
            'Brighton': {
                'name': 'Amex Stadium',
                'capacity': 31800,
                'pitch_quality': 0.9,
                'roof_coverage': 1.0,
                'pitch_size': {'length': 105, 'width': 68},
                'surface_type': 'Desso GrassMaster',
                'drainage_rating': 0.9,
                'floodlight_lux': 1900,
            },
            'Chelsea': {
                'name': 'Stamford Bridge',
                'capacity': 40834,
                'pitch_quality': 0.95,
                'roof_coverage': 0.9,
                'pitch_size': {'length': 103, 'width': 67},
                'surface_type': 'Desso GrassMaster',
                'drainage_rating': 0.9,
                'floodlight_lux': 2100,
            },
            'Crystal Palace': {
                'name': 'Selhurst Park',
                'capacity': 25486,
                'pitch_quality': 0.85,
                'roof_coverage': 0.7,
                'pitch_size': {'length': 101, 'width': 68},
                'surface_type': 'Natural Grass',
                'drainage_rating': 0.8,
                'floodlight_lux': 1600,
            },
            'Everton': {
                'name': 'Goodison Park',
                'capacity': 39572,
                'pitch_quality': 0.85,
                'roof_coverage': 0.8,
                'pitch_size': {'length': 100.48, 'width': 68},
                'surface_type': 'Natural Grass',
                'drainage_rating': 0.8,
                'floodlight_lux': 1700,
            },
            'Liverpool': {
                'name': 'Anfield',
                'capacity': 53394,
                'pitch_quality': 0.95,
                'roof_coverage': 0.9,
                'pitch_size': {'length': 101, 'width': 68},
                'surface_type': 'Desso GrassMaster',
                'drainage_rating': 0.95,
                'floodlight_lux': 2100,
            },
            'Man City': {
                'name': 'Etihad Stadium',
                'capacity': 55097,
                'pitch_quality': 0.95,
                'roof_coverage': 1.0,
                'pitch_size': {'length': 105, 'width': 68},
                'surface_type': 'Desso GrassMaster',
                'drainage_rating': 0.95,
                'floodlight_lux': 2200,
            },
            'Man Utd': {
                'name': 'Old Trafford',
                'capacity': 74140,
                'pitch_quality': 0.9,
                'roof_coverage': 0.95,
                'pitch_size': {'length': 105, 'width': 68},
                'surface_type': 'Desso GrassMaster',
                'drainage_rating': 0.9,
                'floodlight_lux': 2000,
            },
            'Newcastle': {
                'name': 'St. James\' Park',
                'capacity': 52305,
                'pitch_quality': 0.9,
                'roof_coverage': 0.9,
                'pitch_size': {'length': 105, 'width': 68},
                'surface_type': 'Desso GrassMaster',
                'drainage_rating': 0.85,
                'floodlight_lux': 1900,
            },
            'Nottingham Forest': {
                'name': 'City Ground',
                'capacity': 30445,
                'pitch_quality': 0.85,
                'roof_coverage': 0.8,
                'pitch_size': {'length': 102.4, 'width': 68},
                'surface_type': 'Natural Grass',
                'drainage_rating': 0.8,
                'floodlight_lux': 1700,
            },
            'Tottenham': {
                'name': 'Tottenham Hotspur Stadium',
                'capacity': 62850,
                'pitch_quality': 1.0,  # 最新的可分离式草坪系统
                'roof_coverage': 1.0,
                'pitch_size': {'length': 105, 'width': 68},
                'surface_type': 'Hybrid Grass',
                'drainage_rating': 1.0,
                'floodlight_lux': 2500,
            },
            'West Ham': {
                'name': 'London Stadium',
                'capacity': 60000,
                'pitch_quality': 0.9,
                'roof_coverage': 0.95,
                'pitch_size': {'length': 105, 'width': 68},
                'surface_type': 'Desso GrassMaster',
                'drainage_rating': 0.9,
                'floodlight_lux': 2000,
            },
            'Wolves': {
                'name': 'Molineux Stadium',
                'capacity': 32050,
                'pitch_quality': 0.9,
                'roof_coverage': 0.85,
                'pitch_size': {'length': 105, 'width': 68},
                'surface_type': 'Desso GrassMaster',
                'drainage_rating': 0.85,
                'floodlight_lux': 1800,
            },
            'Brentford': {
                'name': 'Gtech Community Stadium',
                'capacity': 17250,
                'pitch_quality': 0.9,
                'roof_coverage': 1.0,
                'pitch_size': {'length': 105, 'width': 68},
                'surface_type': 'Hybrid Grass',
                'drainage_rating': 0.9,
                'floodlight_lux': 1900,
            },
            'Burnley': {
                'name': 'Turf Moor',
                'capacity': 21944,
                'pitch_quality': 0.85,
                'roof_coverage': 0.7,
                'pitch_size': {'length': 105, 'width': 68},
                'surface_type': 'Natural Grass',
                'drainage_rating': 0.8,
                'floodlight_lux': 1600,
            },
            'Luton': {
                'name': 'Kenilworth Road',
                'capacity': 10356,
                'pitch_quality': 0.8,
                'roof_coverage': 0.6,
                'pitch_size': {'length': 100, 'width': 65},
                'surface_type': 'Natural Grass',
                'drainage_rating': 0.75,
                'floodlight_lux': 1500,
            },
            'Sheffield United': {
                'name': 'Bramall Lane',
                'capacity': 32125,
                'pitch_quality': 0.85,
                'roof_coverage': 0.8,
                'pitch_size': {'length': 102, 'width': 66},
                'surface_type': 'Natural Grass',
                'drainage_rating': 0.8,
                'floodlight_lux': 1700,
            }
        }
        
        # 观众因素基准值（基于历史主场优势和球迷氛围）
        self.base_crowd_factors = {
            'Liverpool': 0.95,    # 安菲尔德氛围举世闻名
            'Man Utd': 0.95,     # 老特拉福德容量最大
            'Newcastle': 0.93,    # 圣詹姆斯公园氛围出众
            'Tottenham': 0.92,    # 新球场设计增强声学效果
            'Arsenal': 0.90,
            'Man City': 0.90,
            'Chelsea': 0.90,
            'West Ham': 0.88,
            'Aston Villa': 0.88,
            'Everton': 0.88,
            'Leeds': 0.88,
            'Brighton': 0.85,
            'Crystal Palace': 0.85,
            'Wolves': 0.85,
            'Brentford': 0.85,
            'Nottingham Forest': 0.85,
            'Burnley': 0.83,
            'Sheffield United': 0.83,
            'Luton': 0.82,
        }
    
    def get_venue_data(self, home_team: str, away_team: str, 
                      attendance_percentage: float = None) -> Dict:
        """获取场地数据"""
        stadium = self.stadium_data.get(home_team, {
            'name': f"{home_team} Stadium",
            'capacity': 40000,
            'pitch_quality': 0.85,
            'roof_coverage': 0.7,
        })
        
        # 计算观众因素
        crowd_factor = self._compute_crowd_factor(
            home_team, 
            attendance_percentage or 0.8
        )
        
        # 计算场地状况
        pitch_condition = self._compute_pitch_condition(
            stadium['pitch_quality'],
            stadium['roof_coverage']
        )
        
        return {
            'is_home': True,  # 主场
            'stadium_name': stadium['name'],
            'capacity': stadium['capacity'],
            'pitch_condition': pitch_condition,
            'crowd_factor': crowd_factor
        }
        
    def create_venue_impact(self, home_team: str, away_team: str,
                          match_time: datetime,
                          attendance_percentage: float = None) -> EnvironmentalData:
        """创建场地环境影响数据"""
        venue_data = self.get_venue_data(home_team, away_team, attendance_percentage)
        
        # 计算影响严重程度
        severity = self._compute_venue_severity(venue_data)
        
        return EnvironmentalData(
            source=NewsSource.VENUE,
            timestamp=match_time,
            description=self._generate_venue_description(venue_data),
            severity=severity,
            confidence=0.95,  # 场地数据可信度很高
            event_type='venue_condition',
            venue_data=venue_data
        )
        
    def _compute_crowd_factor(self, team: str, attendance_percentage: float) -> float:
        """计算观众因素"""
        base_factor = self.base_crowd_factors.get(team, 0.8)
        return min(1.0, base_factor * attendance_percentage)
        
    def _compute_pitch_condition(self, base_quality: float, roof_coverage: float) -> float:
        """计算场地状况"""
        # 考虑屋顶覆盖对场地保护的影响
        return min(1.0, base_quality * (0.8 + 0.2 * roof_coverage))
        
    def _compute_venue_severity(self, venue_data: Dict) -> float:
        """计算场地影响严重程度"""
        severity = 0.0
        
        # 场地状况影响
        if venue_data['pitch_condition'] < 0.8:
            severity += (0.8 - venue_data['pitch_condition']) * 0.5
            
        # 观众因素影响
        if venue_data['crowd_factor'] > 0.9:
            severity += (venue_data['crowd_factor'] - 0.9) * 0.5
            
        return min(1.0, severity)
        
    def _generate_venue_description(self, venue_data: Dict) -> str:
        """生成场地描述"""
        desc = f"Match at {venue_data['stadium_name']}, "
        desc += f"Pitch condition: {venue_data['pitch_condition']:.2f}, "
        desc += f"Crowd factor: {venue_data['crowd_factor']:.2f}"
        return desc 