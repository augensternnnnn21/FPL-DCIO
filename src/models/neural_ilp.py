import mip
import torch
import numpy as np

class NeuralILPSolver:
    def __init__(self, budget=100.0, max_per_team=3, total_players=15):
        self.budget = budget
        self.max_per_team = max_per_team
        self.total_players = total_players

    def solve(self, action_probs: torch.Tensor, values: torch.Tensor, teams: np.ndarray, positions=None) -> np.ndarray:
        """
        给定每个球员的选择概率 action_probs、价格 values、球队名称数组 teams、位置数组 positions，
        利用ILP精确选出一个可行解。
        输出: 与 action_probs 维度相同的二进制数组 selected, 表示最终选择结果
        """
        n = len(action_probs)
        # 创建整数线性规划模型
        model = mip.Model()
        # 决策变量x[i]: 是否选择球员 i
        x = [model.add_var(var_type=mip.BINARY) for _ in range(n)]

        # 目标函数: 最大化 Σ(prob[i] * x[i])
        # 可替换为 Σ(prob[i]*得分预测) 或其他优化目标
        model.objective = mip.maximize(
            mip.xsum(float(action_probs[i].item()) * x[i] for i in range(n))
        )

        # 约束1: 总预算 ≤ self.budget
        model.add_constr(
            mip.xsum(float(values[i].item()) * x[i] for i in range(n)) <= self.budget
        )

        # 约束2: 总选人数量 = self.total_players
        model.add_constr(
            mip.xsum(x[i] for i in range(n)) == self.total_players
        )

        # 约束3: 同队球员 ≤ self.max_per_team
        for team in np.unique(teams):
            team_indices = [i for i in range(n) if teams[i] == team]
            if team_indices:
                model.add_constr(
                    mip.xsum(x[i] for i in team_indices) <= self.max_per_team
                )

        # 约束4: 位置要求(如果提供)
        if positions is not None:
            pos_requirements = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            for pos, required in pos_requirements.items():
                pos_indices = [i for i in range(n) if positions[i] == pos]
                if pos_indices:
                    model.add_constr(
                        mip.xsum(x[i] for i in pos_indices) == required
                    )

        # 求解
        status = model.optimize(max_seconds=30) 
        # 这里max_seconds=30仅做演示，可自行调大

        selected = np.zeros(n, dtype=int)
        if status == mip.OptimizationStatus.OPTIMAL:
            for i in range(n):
                if x[i].x >= 0.99:
                    selected[i] = 1

        return selected 