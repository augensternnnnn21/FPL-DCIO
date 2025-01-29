from typing import Dict, List, Optional
import torch
import torch.nn as nn
from dsn_fpl.src.core.chaos.ripple_generator import ChaosRippleGenerator, RippleEvent
from dsn_fpl.src.core.analysis.environment_analyzer import EnvironmentAnalyzer
from dsn_fpl.src.core.nash.game_topology import NashWeaveGraph, Strategy
from dsn_fpl.src.core.quantum.q_player import QPlayer
from dsn_fpl.src.core.data.manager import DataManager
from dsn_fpl.src.core.data.preprocessor import DataPreprocessor

class DSNModel(nn.Module):
    def __init__(self, config: Dict = None, data_manager: DataManager = None):
        super().__init__()
        # 核心组件初始化
        self.ripple_generator = ChaosRippleGenerator()
        self.environment_analyzer = EnvironmentAnalyzer()
        self.nash_graph = NashWeaveGraph()
        self.q_players: Dict[int, QPlayer] = {}
        self.data_manager = data_manager or DataManager()
        
        # 配置初始化
        config = config or {}
        
        # 神经网络层
        self.input_dim = 18
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # 添加批次归一化和更复杂的特征提取器
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 使用双向LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 更复杂的预测器
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),  # *2是因为双向LSTM
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    async def forward(self, batch_data: Dict):
        # 1. 获取时间维度状态
        temporal_state = await self.data_manager.get_temporal_state()
        
        # 2. 环境分析
        env_impacts = self.environment_analyzer.analyze_impacts(
            batch_data['env_data'],
            batch_data['player_ids']
        )
        
        # 3. 生成扰动事件
        for impact in env_impacts.values():
            event = RippleEvent(
                type=impact.sources[0].source,
                intensity=abs(impact.impact_score),
                source_id=str(impact.player_id),
                affected_ids=[str(pid) for pid in batch_data['player_ids']],
                decay_rate=0.1
            )
            self.ripple_generator.add_ripple(event)
        
        # 4. 更新量子状态
        features = []
        for player_id in batch_data['player_ids']:
            if player_id not in self.q_players:
                self.q_players[player_id] = QPlayer(
                    player_id=player_id,
                    name=batch_data['player_names'][player_id],
                    position=batch_data['positions'][player_id],
                    base_stats=batch_data['stats'][player_id]
                )
            
            # 获取扰动影响
            perturbation = self.ripple_generator.get_entity_perturbation(str(player_id))
            
            # 获取球员基础统计数据
            stats = batch_data['stats'][player_id]
            
            # 合并特征，添加默认值处理
            impact = env_impacts.get(player_id)
            impact_score = impact.impact_score if impact is not None else 0.0
            
            # 归一化处理
            form = float(stats.get('form', 0)) / 10.0  # 假设form最大值为10
            fitness = float(stats.get('fitness', 0))  # 已经是0-1之间
            momentum = float(stats.get('momentum', 0)) / 10.0  # 假设momentum最大值为10
            potential = float(stats.get('potential', 0))  # 已经是0-1之间
            
            # 扩展特征
            quantum_state = torch.tensor(self.q_players[player_id]._quantum_state.current_form, dtype=torch.float32)
            perturbation_tensor = torch.tensor(perturbation, dtype=torch.float32)
            other_features = torch.tensor([
                impact_score / 10.0,  # 归一化环境影响
                form,
                fitness,
                momentum,
                potential
            ], dtype=torch.float32)
            
            print(f"Player {player_id} feature dimensions:")
            print(f"- Quantum state: {quantum_state.shape}")
            print(f"- Perturbation: {perturbation_tensor.shape}")
            print(f"- Other features: {other_features.shape}")
            
            player_features = torch.cat([
                quantum_state,
                perturbation_tensor,
                other_features
            ])
            print(f"Total player features shape: {player_features.shape}")
            features.append(player_features)
        
        # 5. 特征提取
        x = torch.stack(features)
        print(f"Features shape after stack: {x.shape}")
        x = x.to(torch.float32)
        
        # 添加批次归一化
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
        
        # 6. 特征提取
        x = self.feature_extractor(x)
        
        # 6. 时序处理
        x = x.unsqueeze(0)  # 添加批次维度
        x, _ = self.lstm(x)
        x = x.squeeze(0)  # 移除批次维度
        
        # 7. 预测
        predictions = self.predictor(x)
        
        # 8. 更新策略图
        strategy = Strategy(
            player_ids=batch_data['player_ids'],
            formation=batch_data['formation'],
            captain_id=batch_data['captain_id'],
            expected_points=predictions.mean().item(),
            ownership_rate=batch_data.get('ownership_rate', 0.5)
        )
        self.nash_graph.add_strategy(strategy)
        
        return predictions
        
    def find_optimal_strategy(self) -> Strategy:
        """找到最优策略"""
        nash_equilibria = self.nash_graph.find_nash_equilibria()
        if not nash_equilibria:
            return None
        return max(nash_equilibria, key=lambda s: s.expected_points)

class DSNTrainer:
    """DSN模型训练器"""
    def __init__(self, model: DSNModel, config: Dict = None):
        self.model = model
        self.config = config or {}
        
        # 优化器使用AdamW
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=0.01
        )
        
        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 使用SmoothL1Loss代替MSE
        self.criterion = nn.SmoothL1Loss()
        
    async def train_step(self, batch_data: Dict):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        predictions = await self.model(batch_data)
        
        # 归一化目标值
        target_points = torch.tensor(batch_data['target_points']).float()
        target_points = (target_points - target_points.mean()) / (target_points.std() + 1e-8)
        
        # 计算损失
        loss = self.criterion(predictions, target_points)
        
        # 添加L2正则化
        l2_lambda = 0.01
        l2_reg = torch.tensor(0.).to(predictions.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新学习率
        self.scheduler.step(loss)
        
        return loss.item()
        
    async def train(self, train_data: List[Dict], epochs: int):
        """完整训练流程"""
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_data:
                loss = await self.train_step(batch)
                total_loss += loss
                
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}')
            
            # 每个epoch后找到最优策略
            optimal_strategy = self.model.find_optimal_strategy()
            if optimal_strategy:
                print(f'\n最优策略:')
                print(f'球员: {optimal_strategy.player_ids}')
                print(f'阵型: {optimal_strategy.formation}')
                print(f'队长: {optimal_strategy.captain_id}')
                print(f'预期得分: {optimal_strategy.expected_points:.2f}')
