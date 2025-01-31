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
        self.input_dim = 55  # 更新为实际特征维度：12(quantum) + 10(perturbation) + 4(weather) + 3(media) + 2(venue) + 5(other) + 19(base)
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
            
            # 获取天气数据
            weather_features = self._process_weather_data(
                batch_data['weather_data'].get(player_id, {})
            )
            
            # 获取媒体数据
            media_features = self._process_media_data(
                batch_data['media_data'].get(player_id, [])
            )
            
            # 获取场地数据
            venue_features = self._process_venue_data(
                batch_data['venue_data'].get(player_id, {})
            )
            
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
            
            # 基础统计特征
            base_features = torch.tensor([
                float(stats.get('minutes', 0)) / 90.0,  # 归一化出场时间
                float(stats.get('total_points', 0)) / 20.0,  # 归一化总分
                float(stats.get('goals_scored', 0)) / 3.0,  # 归一化进球数
                float(stats.get('assists', 0)) / 3.0,  # 归一化助攻数
                float(stats.get('clean_sheets', 0)) / 1.0,  # 归一化零封数
                float(stats.get('goals_conceded', 0)) / 3.0,  # 归一化失球数
                float(stats.get('own_goals', 0)) / 1.0,  # 归一化乌龙球数
                float(stats.get('penalties_saved', 0)) / 1.0,  # 归一化扑点数
                float(stats.get('penalties_missed', 0)) / 1.0,  # 归一化罚失点球数
                float(stats.get('yellow_cards', 0)) / 1.0,  # 归一化黄牌数
                float(stats.get('red_cards', 0)) / 1.0,  # 归一化红牌数
                float(stats.get('saves', 0)) / 5.0,  # 归一化扑救数
                float(stats.get('bonus', 0)) / 3.0,  # 归一化奖励分
                float(stats.get('bps', 0)) / 1000.0,  # 归一化BPS
                float(stats.get('influence', 0)) / 100.0,  # 归一化影响力
                float(stats.get('creativity', 0)) / 100.0,  # 归一化创造力
                float(stats.get('threat', 0)) / 100.0,  # 归一化威胁值
                float(stats.get('ict_index', 0)) / 100.0,  # 归一化ICT指数
                float(stats.get('selected_by_percent', 0)) / 100.0  # 归一化选择率
            ], dtype=torch.float32)
            
            # 合并所有特征
            player_features = torch.cat([
                quantum_state,
                perturbation_tensor,
                weather_features,
                media_features,
                venue_features,
                torch.tensor([impact_score, form, fitness, momentum, potential], dtype=torch.float32),
                base_features
            ])
            
            # 打印特征维度
            print(f"Feature dimensions:")
            print(f"quantum_state: {quantum_state.shape}")
            print(f"perturbation_tensor: {perturbation_tensor.shape}")
            print(f"weather_features: {weather_features.shape}")
            print(f"media_features: {media_features.shape}")
            print(f"venue_features: {venue_features.shape}")
            print(f"other_features: {torch.tensor([impact_score, form, fitness, momentum, potential], dtype=torch.float32).shape}")
            print(f"base_features: {base_features.shape}")
            print(f"Total features: {player_features.shape}")
            
            features.append(player_features)
        
        # 5. 特征提取
        x = torch.stack(features)
        x = x.to(torch.float32)
        
        # 添加批次归一化
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
        
        # 6. 特征提取
        x = self.feature_extractor(x)
        
        # 7. 时序处理
        x = x.unsqueeze(0)  # 添加批次维度
        x, _ = self.lstm(x)
        x = x.squeeze(0)  # 移除批次维度
        
        # 8. 预测
        predictions = self.predictor(x)
        
        # 9. 更新策略图
        strategy = Strategy(
            player_ids=batch_data['player_ids'],
            formation=batch_data['formation'],
            captain_id=batch_data['captain_id'],
            expected_points=predictions.mean().item(),
            ownership_rate=batch_data.get('ownership_rate', 0.5)
        )
        self.nash_graph.add_strategy(strategy)
        
        return predictions
        
    def _process_weather_data(self, weather_data: Dict) -> torch.Tensor:
        """处理天气数据"""
        if not weather_data:
            return torch.zeros(4)  # 默认特征维度
            
        features = [
            weather_data.get('temperature', 20) / 40,  # 归一化温度
            weather_data.get('precipitation', 0),  # 已经是0-1
            weather_data.get('wind_speed', 0) / 10,  # 归一化风速
            weather_data.get('humidity', 50) / 100  # 归一化湿度
        ]
        
        return torch.tensor(features, dtype=torch.float32)
        
    def _process_media_data(self, media_articles: List[Dict]) -> torch.Tensor:
        """处理媒体数据"""
        if not media_articles:
            return torch.zeros(3)  # 默认特征维度
            
        # 计算平均情感得分
        sentiment_scores = [
            article['sentiment']['score']
            for article in media_articles
        ]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # 计算报道强度
        coverage_intensity = min(len(media_articles) / 10, 1.0)
        
        # 计算平均置信度
        confidence = sum(
            article['sentiment']['confidence']
            for article in media_articles
        ) / len(media_articles)
        
        features = [
            avg_sentiment,  # 已经是-1到1
            coverage_intensity,  # 0到1
            confidence  # 0到1
        ]
        
        return torch.tensor(features, dtype=torch.float32)
        
    def _process_venue_data(self, venue_data: Dict) -> torch.Tensor:
        """处理场地数据"""
        if not venue_data:
            return torch.zeros(2)  # 更新特征维度为2
            
        # 计算场地特征
        capacity_normalized = venue_data.get('capacity', 40000) / 75000  # 归一化容量
        attendance_ratio = venue_data.get('average_attendance', 30000) / venue_data.get('capacity', 40000)
        
        features = [
            capacity_normalized,  # 容量 (0-1)
            attendance_ratio,  # 上座率 (0-1)
        ]
        
        return torch.tensor(features, dtype=torch.float32)
        
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
