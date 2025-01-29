from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
from ..core.temporal.origami_planner import TemporalOrigamiPlanner
from ..core.data.manager import DataManager
from .metrics import ModelEvaluator

class Backtester:
    """回测器"""
    def __init__(self,
                 historical_data: pd.DataFrame,
                 start_gw: int,
                 end_gw: int):
        self.historical_data = historical_data
        self.start_gw = start_gw
        self.end_gw = end_gw
        self.evaluator = ModelEvaluator()
        
    async def run_backtest(self,
                          initial_team: List[int],
                          budget: float,
                          transfers_per_gw: int = 1) -> Dict:
        """运行回测"""
        results = {
            'predictions': [],
            'actual_outcomes': [],
            'metrics': []
        }
        
        current_team = initial_team.copy()
        remaining_budget = budget
        
        for gw in range(self.start_gw, self.end_gw + 1):
            # 获取当前gameweek的数据
            gw_data = self._get_gameweek_data(gw)
            
            # 生成预测
            predictions = await self._generate_predictions(
                current_team,
                remaining_budget,
                transfers_per_gw,
                gw_data
            )
            
            # 记录实际结果
            actual_outcomes = self._get_actual_outcomes(gw + 1, predictions)
            
            # 评估预测
            metrics = self.evaluator.evaluate_predictions(
                predictions,
                actual_outcomes['points'],
                actual_outcomes['timing'],
                budget - remaining_budget
            )
            
            # 更新团队和预算
            current_team, remaining_budget = self._apply_transfers(
                current_team,
                remaining_budget,
                predictions,
                gw_data
            )
            
            # 记录结果
            results['predictions'].append(predictions)
            results['actual_outcomes'].append(actual_outcomes)
            results['metrics'].append(metrics)
            
        return results
        
    def _get_gameweek_data(self, gw: int) -> pd.DataFrame:
        """获取特定gameweek的数据"""
        return self.historical_data[self.historical_data['gw'] == gw]
        
    async def _generate_predictions(self,
                                  current_team: List[int],
                                  budget: float,
                                  num_transfers: int,
                                  gw_data: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """生成预测"""
        planner = TemporalOrigamiPlanner()
        data_manager = DataManager()
        
        # 获取时间维度状态
        temporal_state = await data_manager.get_temporal_state()
        
        # 生成转会建议
        recommendations = planner.generate_transfer_plan(
            current_team,
            budget,
            num_transfers
        )
        
        return [
            (old_id, new_id, 0.8)  # 简化的置信度
            for old_id, new_id, _ in recommendations
        ]
        
    def _get_actual_outcomes(self,
                           next_gw: int,
                           predictions: List[Tuple[int, int, float]]) -> Dict:
        """获取实际结果"""
        next_gw_data = self._get_gameweek_data(next_gw)
        
        return {
            'points': {
                player_id: next_gw_data[
                    next_gw_data['player_id'] == player_id
                ]['total_points'].iloc[0]
                for _, player_id, _ in predictions
            },
            'timing': {
                player_id: datetime.now()  # 简化的时机记录
                for _, player_id, _ in predictions
            }
        }
        
    def _apply_transfers(self,
                        team: List[int],
                        budget: float,
                        predictions: List[Tuple[int, int, float]],
                        gw_data: pd.DataFrame) -> Tuple[List[int], float]:
        """应用转会决策"""
        new_team = team.copy()
        remaining_budget = budget
        
        for old_id, new_id, _ in predictions:
            if old_id in new_team:
                # 计算转会费用
                old_price = gw_data[
                    gw_data['player_id'] == old_id
                ]['value'].iloc[0]
                new_price = gw_data[
                    gw_data['player_id'] == new_id
                ]['value'].iloc[0]
                
                cost = new_price - old_price
                if cost <= remaining_budget:
                    new_team.remove(old_id)
                    new_team.append(new_id)
                    remaining_budget -= cost
                    
        return new_team, remaining_budget 