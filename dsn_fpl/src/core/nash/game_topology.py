from typing import Dict, List, Set, Tuple
import networkx as nx
import numpy as np
from dataclasses import dataclass

@dataclass
class Strategy:
    """策略表示"""
    player_ids: List[int]
    formation: str
    captain_id: int
    expected_points: float
    ownership_rate: float

class NashWeaveGraph:
    def __init__(self):
        self.strategy_graph = nx.DiGraph()
        self.nash_equilibria: List[Strategy] = []
        self.crowd_density_map: Dict[int, float] = {}
        
    def add_strategy(self, strategy: Strategy):
        """添加策略节点到博弈图"""
        node_id = hash(tuple(strategy.player_ids))
        self.strategy_graph.add_node(
            node_id,
            strategy=strategy,
            crowd_density=strategy.ownership_rate
        )
        self._update_edges(node_id, strategy)
        
    def _update_edges(self, node_id: int, strategy: Strategy):
        """更新策略间的博弈关系边"""
        for other_node in self.strategy_graph.nodes():
            if other_node != node_id:
                other_strategy = self.strategy_graph.nodes[other_node]['strategy']
                payoff = self._calculate_payoff(strategy, other_strategy)
                self.strategy_graph.add_edge(node_id, other_node, weight=payoff)
                
    def _calculate_payoff(self, strategy1: Strategy, strategy2: Strategy) -> float:
        """计算两个策略间的博弈收益"""
        common_players = set(strategy1.player_ids) & set(strategy2.player_ids)
        uniqueness_factor = 1 - len(common_players) / len(strategy1.player_ids)
        expected_points_diff = strategy1.expected_points - strategy2.expected_points
        ownership_advantage = 1 - strategy1.ownership_rate
        
        return uniqueness_factor * expected_points_diff * ownership_advantage
    
    def find_nash_equilibria(self) -> List[Strategy]:
        """寻找纳什均衡策略"""
        self.nash_equilibria = []
        payoff_matrix = nx.adjacency_matrix(self.strategy_graph).todense()
        
        # 使用Lemke-Howson算法寻找纳什均衡
        n = len(payoff_matrix)
        for i in range(n):
            if self._is_best_response(i, payoff_matrix):
                strategy = self.strategy_graph.nodes[list(self.strategy_graph.nodes())[i]]['strategy']
                self.nash_equilibria.append(strategy)
                
        return self.nash_equilibria
    
    def _is_best_response(self, strategy_idx: int, payoff_matrix: np.ndarray) -> bool:
        """判断是否为最优响应策略"""
        payoffs = payoff_matrix[strategy_idx, :]
        return np.all(payoffs >= np.max(payoff_matrix, axis=0))
    
    def find_anti_consensus_strategies(self, threshold: float = 0.2) -> List[Strategy]:
        """发现反共识策略"""
        anti_consensus = []
        for node in self.strategy_graph.nodes():
            strategy = self.strategy_graph.nodes[node]['strategy']
            if (strategy.ownership_rate < threshold and 
                strategy.expected_points > np.mean([s.expected_points for s in self.nash_equilibria])):
                anti_consensus.append(strategy)
        return anti_consensus
    
    def update_crowd_density(self, ownership_data: Dict[int, float]):
        """更新群体密度图"""
        self.crowd_density_map = ownership_data
        for node in self.strategy_graph.nodes():
            strategy = self.strategy_graph.nodes[node]['strategy']
            avg_ownership = np.mean([ownership_data.get(pid, 0) for pid in strategy.player_ids])
            self.strategy_graph.nodes[node]['crowd_density'] = avg_ownership 