# src/old/benchmark.py

import torch
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from models import DCIO
from environment import FPLEnv
from config import *
import pandas as pd
import os

def run_benchmarks():
    """运行完整的基准测试"""
    results = {}
    
    print("开始基准测试...")
    
    # 创建环境实例
    test_seasons = ["2019-20", "2020-21"]  # 测试多个赛季
    for season in test_seasons:
        print(f"\n测试赛季: {season}")
        env = FPLEnv(season=season)
        
        # 1. 贪心策略 - 基于性价比的简单选择
        print("\n评估贪心策略...")
        results['Greedy'] = evaluate_greedy_strategy(env)
        
        # 2. 动态规划 - 考虑预算约束的最优选择
        print("\n评估动态规划策略...")
        results['DynamicProgramming'] = evaluate_dp_strategy(env)
        
        # 3. 遗传算法 - 处理多约束组合优化
        print("\n评估遗传算法策略...")
        results['GeneticAlgorithm'] = evaluate_ga_strategy(env)
        
        # 4. MCTS - 考虑长期收益的决策
        print("\n评估MCTS策略...")
        results['MCTS'] = evaluate_mcts_strategy(env)
        
        # 5. 我们的模型
        print("\n评估DCIO模型...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results['DCIO'] = evaluate_our_model(env, device)
        
        # 添加MAML模型评估
        print("\n评估MAML模型...")
        results[f'MAML_{season}'] = evaluate_maml_model(env)
    
    # 打印结果
    print("\n基准测试结果:")
    print("-" * 60)
    print(f"{'策略':<20} | {'平均得分':<15} | {'标准差':<15}")
    print("-" * 60)
    for strategy, (mean, std) in results.items():
        print(f"{strategy:<20} | {mean:>15.1f} | {std:>15.1f}")
    print("-" * 60)
    
    return results

def evaluate_greedy_strategy(env, episodes=100):
    """贪心策略：基于性价比选择球员"""
    total_points = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        last_points = 0
        
        while not done:
            data = env.data.copy()
            
            # 确保数值类型
            data['total_points'] = pd.to_numeric(data['total_points'], errors='coerce')
            data['value'] = pd.to_numeric(data['value'], errors='coerce')
            data['3_gw_avg'] = pd.to_numeric(data['3_gw_avg'], errors='coerce')
            
            # 添加随机扰动
            random_noise = np.random.normal(0, 0.1, len(data))
            
            # 计算性价比分数
            data['points_per_cost'] = data['total_points'] / data['value'].clip(1)
            data['recent_form'] = data['3_gw_avg'].fillna(0)
            data['efficiency_score'] = (data['points_per_cost'] * 0.7 + 
                                      data['recent_form'] * 0.3 + 
                                      random_noise)  # 添加随机性
            
            selected_players = []
            remaining_budget = BUDGET
            
            # 2-5-5-3阵型
            formation = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            
            # 按位置贪心选择
            for pos, required_count in formation.items():
                pos_players = data[data['position'] == pos].copy()
                pos_players = pos_players[pos_players['value'] <= remaining_budget]
                
                # 从top N个球员中随机选择
                top_n = min(required_count * 3, len(pos_players))
                if top_n < required_count:
                    print(f"警告：{pos}位置可选球员不足")
                    break
                    
                top_players = pos_players.nlargest(top_n, 'efficiency_score')
                selected_indices = np.random.choice(top_players.index, 
                                                  size=required_count, 
                                                  replace=False)
                
                for idx in selected_indices:
                    selected_players.append(idx)
                    remaining_budget -= data.loc[idx, 'value']
            
            # 执行动作
            if len(selected_players) == 15:
                action = np.array(selected_players)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                last_points = sum(data.loc[action, 'total_points'])
            else:
                print(f"警告：只选择了{len(selected_players)}名球员")
                done = True
                last_points = 0
            
            state = next_state
        
        total_points.append(last_points)
        
        if (ep + 1) % 10 == 0:
            print(f"贪心策略评估进度: {ep + 1}/{episodes}, "
                  f"当前平均得分: {np.mean(total_points):.1f}")
    
    return np.mean(total_points), np.std(total_points)

def evaluate_dp_strategy(env, episodes=100):
    """动态规划策略：考虑预算约束的最优选择"""
    
    def solve_knapsack(players, budget, required_count):
        """解决背包问题的贪心近似解"""
        # 预处理：只考虑预算内的球员
        players = players[players['value'] <= budget].copy()
        if len(players) < required_count:
            return []
        
        # 计算性价比
        players['efficiency'] = players['total_points'] / players['value'].clip(1)
        
        # 按性价比排序选择球员
        selected = players.nlargest(required_count, 'efficiency').index.tolist()
        
        return selected

    total_points = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        last_points = 0
        
        while not done:
            data = env.data.copy()
            # 确保数值类型
            data['total_points'] = pd.to_numeric(data['total_points'], errors='coerce')
            data['value'] = pd.to_numeric(data['value'], errors='coerce')
            
            selected_players = []
            remaining_budget = BUDGET
            
            # 按位置选择
            formation = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            
            # 添加随机性
            noise = np.random.normal(0, 0.1, len(data))
            data['random_factor'] = noise
            
            for pos, required_count in formation.items():
                pos_players = data[data['position'] == pos].copy()
                if len(pos_players) < required_count:
                    print(f"警告：{pos}位置可选球员不足")
                    break
                    
                # 考虑预算约束
                available_players = pos_players[pos_players['value'] <= remaining_budget]
                if len(available_players) < required_count:
                    print(f"警告：{pos}位置预算内球员不足")
                    break
                
                # 选择球员
                pos_selected = solve_knapsack(available_players, remaining_budget, required_count)
                if len(pos_selected) < required_count:
                    print(f"警告：{pos}位置无法选择足够的球员")
                    break
                    
                selected_players.extend(pos_selected)
                remaining_budget -= pos_players.loc[pos_selected, 'value'].sum()
            
            # 执行动作
            if len(selected_players) == 15:
                action = np.array(selected_players)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                last_points = sum(data.loc[action, 'total_points'])
            else:
                print(f"警告：只选择了{len(selected_players)}名球员")
                done = True
                last_points = 0
            
            state = next_state
        
        total_points.append(last_points)
        
        if (ep + 1) % 10 == 0:
            print(f"动态规划评估进度: {ep + 1}/{episodes}, "
                  f"当前平均得分: {np.mean(total_points):.1f}")
    
    return np.mean(total_points), np.std(total_points)

def evaluate_ga_strategy(env, episodes=100, population_size=100, generations=50):
    """遗传算法策略：使用遗传算法优化球队选择"""
    
    def create_individual():
        """创建一个合法的个体（球队阵容）"""
        selected = []
        remaining_budget = BUDGET
        formation = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        for pos, required_count in formation.items():
            pos_players = env.data[env.data['position'] == pos].copy()
            available_players = pos_players[pos_players['value'] <= remaining_budget]
            
            if len(available_players) >= required_count:
                selected_players = available_players.sample(n=required_count).index
                selected.extend(selected_players)
                remaining_budget -= pos_players.loc[selected_players, 'value'].sum()
        
        return np.array(selected)
    
    def fitness(individual):
        """计算适应度（预期得分）"""
        if len(individual) != 15:
            return -1000  # 惩罚非法解
        
        total_value = env.data.loc[individual, 'value'].sum()
        if total_value > BUDGET:
            return -1000  # 惩罚超预算
        
        # 计算得分
        points = env.data.loc[individual, 'total_points'].sum()
        form_score = env.data.loc[individual, '3_gw_avg'].mean()
        
        return points + form_score * 0.3  # 综合考虑总分和近期表现
    
    def crossover(parent1, parent2):
        """交叉操作"""
        # 按位置进行交叉
        child = []
        formation = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        for pos, count in formation.items():
            pos_mask = env.data['position'] == pos
            p1_players = [p for p in parent1 if pos_mask[p]]
            p2_players = [p for p in parent2 if pos_mask[p]]
            
            # 随机选择父母的球员
            selected = []
            for i in range(count):
                if np.random.random() < 0.5 and i < len(p1_players):
                    selected.append(p1_players[i])
                elif i < len(p2_players):
                    selected.append(p2_players[i])
            
            child.extend(selected)
        
        return np.array(child)
    
    def mutate(individual, mutation_rate=0.1):
        """变异操作"""
        if np.random.random() > mutation_rate:
            return individual
        
        # 随机选择一个位置进行变异
        pos = np.random.choice(['GK', 'DEF', 'MID', 'FWD'])
        pos_mask = env.data['position'] == pos
        current_players = [p for p in individual if pos_mask[p]]
        
        if len(current_players) > 0:
            # 随机替换一名球员
            player_to_replace = np.random.choice(current_players)
            available_players = env.data[pos_mask].index
            new_player = np.random.choice(available_players)
            
            individual = np.array([new_player if p == player_to_replace else p for p in individual])
        
        return individual
    
    total_points = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        last_points = 0
        
        # 创建初始种群
        population = [create_individual() for _ in range(population_size)]
        
        while not done:
            # 进行遗传算法优化
            for gen in range(generations):
                # 计算适应度
                fitness_scores = [fitness(ind) for ind in population]
                
                # 选择
                selected_indices = np.argsort(fitness_scores)[-population_size//2:]
                parents = [population[i] for i in selected_indices]
                
                # 生成新一代
                new_population = parents.copy()
                while len(new_population) < population_size:
                    p1, p2 = np.random.choice(parents, 2, replace=False)
                    child = crossover(p1, p2)
                    child = mutate(child)
                    new_population.append(child)
                
                population = new_population
            
            # 选择最优个体
            best_individual = population[np.argmax([fitness(ind) for ind in population])]
            
            # 执行动作
            action = best_individual
            if len(action) == 15:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                last_points = sum(env.data.loc[action, 'total_points'])
            else:
                print(f"警告：遗传算法未能找到合法解")
                done = True
                last_points = 0
            
            state = next_state
        
        total_points.append(last_points)
        
        if (ep + 1) % 10 == 0:
            print(f"遗传算法评估进度: {ep + 1}/{episodes}, "
                  f"当前平均得分: {np.mean(total_points):.1f}")
    
    return np.mean(total_points), np.std(total_points)

def evaluate_model(model, env, device, episodes=100):
    """评估模型性能"""
    model.eval()
    total_points = []
    
    with torch.no_grad():
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            last_points = 0  # 只记录最后一步的得分
            
            while not done:
                state_tensor = torch.FloatTensor(state).to(device)
                if len(state_tensor.shape) == 2:
                    state_tensor = state_tensor.unsqueeze(0)
                
                action = model(state_tensor).squeeze().topk(15).indices.cpu().numpy()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 计算当前选择的球员得分
                selected_points = env.data.loc[action, 'total_points'].sum()
                last_points = selected_points  # 更新最后一步的得分
                
                state = next_state
            
            total_points.append(last_points)  # 只添加最后一步的得分
            
            if (ep + 1) % 10 == 0:
                print(f"评估进度: {ep + 1}/{episodes}, "
                      f"当前平均得分: {np.mean(total_points):.1f}")
    
    return np.mean(total_points), np.std(total_points)

def evaluate_mcts_strategy(env, episodes=100, simulations=100, exploration_constant=1.0):
    """蒙特卡洛树搜索策略"""
    
    class MCTSNode:
        def __init__(self, state, parent=None, action=None):
            self.state = state
            self.parent = parent
            self.action = action
            self.children = {}
            self.visits = 0
            self.value = 0
            self.untried_actions = None
            self.valid_actions = None
    
    def get_valid_actions(state):
        """获取所有合法的动作（球员组合）"""
        data = env.data.copy()
        valid_actions = []
        formation = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        # 按性价比排序
        data['efficiency'] = data['total_points'] / data['value'].clip(1)
        
        # 为每个位置选择top N个球员，减少搜索空间
        position_candidates = {}
        for pos in formation:
            pos_players = data[data['position'] == pos].nlargest(10, 'efficiency')
            position_candidates[pos] = pos_players.index.tolist()
        
        # 生成有效的组合
        for _ in range(50):  # 限制组合数量
            selected = []
            remaining_budget = BUDGET
            valid = True
            
            for pos, required_count in formation.items():
                candidates = position_candidates[pos]
                if len(candidates) < required_count:
                    valid = False
                    break
                    
                # 随机选择指定数量的球员
                selected_players = np.random.choice(candidates, required_count, replace=False)
                player_values = data.loc[selected_players, 'value'].sum()
                
                if player_values <= remaining_budget:
                    selected.extend(selected_players)
                    remaining_budget -= player_values
                else:
                    valid = False
                    break
            
            if valid and len(selected) == 15:
                valid_actions.append(np.array(selected))
        
        return valid_actions
    
    def select_node(node):
        """选择最有潜力的节点"""
        while node.untried_actions is None or (len(node.untried_actions) == 0 and len(node.children) > 0):
            if len(node.children) == 0:
                return node
            
            # UCB1公式选择最佳子节点
            ucb_values = []
            for child in node.children.values():
                if child.visits == 0:
                    ucb_value = float('inf')
                else:
                    exploit = child.value / child.visits
                    explore = exploration_constant * np.sqrt(np.log(node.visits) / child.visits)
                    ucb_value = exploit + explore
                ucb_values.append((child, ucb_value))
            
            node = max(ucb_values, key=lambda x: x[1])[0]
        
        return node
    
    def expand_node(node):
        """扩展节点"""
        if node.untried_actions is None:
            node.untried_actions = get_valid_actions(node.state)
            node.valid_actions = node.untried_actions.copy()
        
        if len(node.untried_actions) > 0:
            action = node.untried_actions.pop()
            next_state, _, _, _, _ = env.step(action)
            child = MCTSNode(next_state, parent=node, action=action)
            node.children[str(action.tolist())] = child
            return child
        
        return node
    
    def simulate(state):
        """模拟到终止状态"""
        total_reward = 0
        done = False
        current_state = state
        
        while not done:
            # 使用简单的贪心策略进行模拟
            data = env.data.copy()
            data['efficiency'] = data['total_points'] / data['value'].clip(1)
            
            selected_players = []
            remaining_budget = BUDGET
            formation = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            
            for pos, required_count in formation.items():
                pos_players = data[data['position'] == pos].copy()
                available = pos_players[pos_players['value'] <= remaining_budget]
                if len(available) >= required_count:
                    selected = available.nlargest(required_count, 'efficiency').index
                    selected_players.extend(selected)
                    remaining_budget -= pos_players.loc[selected, 'value'].sum()
            
            if len(selected_players) == 15:
                action = np.array(selected_players)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += sum(data.loc[action, 'total_points'])
                current_state = next_state
            else:
                break
        
        return total_reward
    
    def backpropagate(node, reward):
        """反向传播更新节点统计信息"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    total_points = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        last_points = 0
        
        while not done:
            # 创建根节点
            root = MCTSNode(state)
            
            # 进行MCTS模拟
            for _ in range(simulations):
                # 选择
                node = select_node(root)
                # 扩展
                node = expand_node(node)
                # 模拟
                reward = simulate(node.state)
                # 反向传播
                backpropagate(node, reward)
            
            # 选择访问次数最多的动作
            if len(root.children) > 0:
                best_child = max(root.children.values(), key=lambda x: x.visits)
                action = best_child.action
            else:
                # 如果没有子节点，使用贪心策略
                action = get_valid_actions(state)[0]
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            last_points = sum(env.data.loc[action, 'total_points'])
            state = next_state
        
        total_points.append(last_points)
        
        if (ep + 1) % 10 == 0:
            print(f"MCTS评估进度: {ep + 1}/{episodes}, "
                  f"当前平均得分: {np.mean(total_points):.1f}")
    
    return np.mean(total_points), np.std(total_points)

def evaluate_maml_model(env, episodes=100, inner_steps=5, inner_lr=0.01):
    """使用MAML评估模型的泛化能力"""
    
    class MAMLModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=128):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)  # 输出球员的价值评分
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        
        def clone(self):
            """创建模型的深度复制"""
            clone = MAMLModel(self.fc1.in_features)
            clone.load_state_dict(self.state_dict())
            return clone
    
    def inner_loop_update(model, support_data, inner_lr):
        """MAML内循环更新"""
        # 创建模型副本
        updated_model = model.clone()
        optimizer = torch.optim.SGD(updated_model.parameters(), lr=inner_lr)
        
        # 在support set上更新模型
        features = torch.FloatTensor(support_data['features'])
        targets = torch.FloatTensor(support_data['targets'])
        
        predictions = updated_model(features)
        loss = F.mse_loss(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return updated_model
    
    def prepare_batch_data(env):
        """准备批量数据"""
        data = env.data.copy()
        
        # 准备特征
        features = []
        for _, player in data.iterrows():
            feat = [
                player['value'],
                player['total_points'],
                player['minutes'],
                player['3_gw_avg'],
                player['is_injured'],
                pd.Categorical(player['position']).codes[0],
                pd.Categorical(player['team']).codes[0]
            ]
            features.append(feat)
        
        # 准备目标值（使用总得分作为目标）
        targets = data['total_points'].values
        
        return {
            'features': np.array(features, dtype=np.float32),
            'targets': targets.reshape(-1, 1)
        }
    
    # 初始化MAML模型
    input_dim = 7  # 特征维度
    model = MAMLModel(input_dim).to(device)
    
    # 加载预训练的模型权重
    if os.path.exists('models/maml_model.pth'):
        model.load_state_dict(torch.load('models/maml_model.pth'))
    
    total_points = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        last_points = 0
        
        # 准备当前环境的数据
        batch_data = prepare_batch_data(env)
        
        # MAML内循环适应
        for _ in range(inner_steps):
            model = inner_loop_update(model, batch_data, inner_lr)
        
        while not done:
            # 使用适应后的模型进行预测
            data = env.data.copy()
            features = torch.FloatTensor(prepare_batch_data(env)['features']).to(device)
            
            with torch.no_grad():
                player_values = model(features).cpu().numpy()
            
            # 选择最佳球员组合
            selected_players = []
            remaining_budget = BUDGET
            formation = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            
            for pos, required_count in formation.items():
                pos_mask = data['position'] == pos
                pos_players = data[pos_mask].copy()
                pos_players['predicted_value'] = player_values[pos_mask]
                
                # 考虑预算约束
                available = pos_players[pos_players['value'] <= remaining_budget]
                if len(available) >= required_count:
                    selected = available.nlargest(required_count, 'predicted_value').index
                    selected_players.extend(selected)
                    remaining_budget -= pos_players.loc[selected, 'value'].sum()
            
            # 执行动作
            if len(selected_players) == 15:
                action = np.array(selected_players)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                last_points = sum(data.loc[action, 'total_points'])
            else:
                print(f"警告：只选择了{len(selected_players)}名球员")
                done = True
                last_points = 0
            
            state = next_state
        
        total_points.append(last_points)
        
        if (ep + 1) % 10 == 0:
            print(f"MAML评估进度: {ep + 1}/{episodes}, "
                  f"当前平均得分: {np.mean(total_points):.1f}")
    
    return np.mean(total_points), np.std(total_points)

if __name__ == "__main__":
    results = run_benchmarks()