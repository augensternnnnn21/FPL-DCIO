from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel

app = FastAPI()

class TransferRequest(BaseModel):
    team_ids: List[int]
    budget: float
    num_transfers: int

class EnvironmentRequest(BaseModel):
    player_ids: List[int]
    location: str
    match_time: datetime

@app.post("/api/v1/analyze/environment")
async def analyze_environment(request: EnvironmentRequest):
    """分析环境影响"""
    try:
        collector = EnvironmentalDataCollector(config)
        analyzer = EnvironmentAnalyzer()
        
        # 收集环境数据
        weather = await collector.get_weather_data(
            request.location, 
            request.match_time
        )
        news = await collector.get_news_sentiment(request.location)
        
        # 分析影响
        impacts = analyzer.analyze_impacts(
            [weather] + news,
            request.player_ids
        )
        
        return {"impacts": impacts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/recommend/transfers")
async def recommend_transfers(request: TransferRequest):
    """生成转会建议"""
    try:
        planner = TemporalOrigamiPlanner()
        
        # 获取时间维度状态
        data_manager = DataManager()
        temporal_state = await data_manager.get_temporal_state()
        
        # 生成建议
        recommendations = planner.generate_transfer_plan(
            request.team_ids,
            request.budget,
            request.num_transfers
        )
        
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 