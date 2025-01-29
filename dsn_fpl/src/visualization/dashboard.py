import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

class DashboardVisualizer:
    """仪表盘可视化器"""
    def __init__(self):
        st.set_page_config(layout="wide")
        self.colors = {
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6'
        }
        
    def render_environmental_impacts(self, impacts: Dict[int, EnvironmentalImpact]):
        """渲染环境影响分析"""
        st.header("环境影响分析")
        
        # 转换数据为DataFrame
        df = pd.DataFrame([
            {
                'player_id': impact.player_id,
                'impact_score': impact.impact_score,
                'confidence': impact.confidence,
                'sources_count': len(impact.sources)
            }
            for impact in impacts.values()
        ])
        
        # 创建影响力热图
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=df['impact_score'].values.reshape(-1, 1),
            x=['Impact'],
            y=df['player_id'],
            colorscale='RdYlGn',
            text=df['impact_score'].round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=True
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_temporal_analysis(self, temporal_states: List[TemporalState]):
        """渲染时间维度分析"""
        st.header("时间维度分析")
        
        # 创建时间序列图
        df = pd.DataFrame([
            {
                'timestamp': state.timestamp,
                'fixture_difficulty': np.mean(list(state.fixture_difficulty.values())),
                'fatigue_index': np.mean(list(state.fatigue_index.values())),
                'form_trajectory': np.mean([t[-1] for t in state.form_trajectory.values()])
            }
            for state in temporal_states
        ])
        
        fig = px.line(df, x='timestamp', y=['fixture_difficulty', 'fatigue_index', 'form_trajectory'],
                     title='Temporal Trends')
        st.plotly_chart(fig, use_container_width=True)
        
    def render_decision_support(self, 
                              recommendations: List[Tuple[int, int, TimeScale]],
                              player_data: Dict[int, Dict]):
        """渲染决策支持视图"""
        st.header("转会建议")
        
        for old_id, new_id, timing in recommendations:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("转出球员")
                st.write(player_data[old_id]['name'])
                
            with col2:
                st.subheader("建议引入")
                st.write(player_data[new_id]['name'])
                
            with col3:
                st.subheader("最佳时机")
                st.write(timing.value) 