# dsn_fpl/scripts/train_dsn.py

import asyncio
import torch
from torch import nn, optim
from pathlib import Path
from dsn_fpl.src.models.dsn_model import DSNModel, DSNTrainer
from dsn_fpl.src.core.data.dataset_generator import DSNDatasetGenerator
from dsn_fpl.src.core.data.manager import DataManager

async def train_model(epochs: int = 50):
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data"
    print(f"Loading data from: {data_path}")
    
    # 初始化数据生成器
    generator = DSNDatasetGenerator(data_path)
    historical_data = generator.load_historical_data()
    
    # 初始化模型
    model = DSNModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nEpoch 1/{epochs}")
    for epoch in range(epochs):
        try:
            # 生成训练批次
            batch = await generator.generate_training_batch(historical_data)
            
            # 前向传播
            predictions = await model(batch)
            targets = torch.tensor(batch['target_points'], dtype=torch.float32)
            
            # 计算损失
            loss = criterion(predictions, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"训练过程中出错: {str(e)}")
            continue
            
    # 关闭资源
    await model.close()
    await generator.media_collector.close()

async def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 初始化数据生成器
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data"
    print(f"Loading data from: {data_path}")
    
    dataset_generator = DSNDatasetGenerator(data_path)
    
    # 加载历史数据
    historical_data = dataset_generator.load_historical_data()
    
    # 初始化数据管理器
    data_manager = DataManager(historical_data)
    
    # 初始化模型和训练器
    model = DSNModel(data_manager=data_manager)
    trainer = DSNTrainer(model)
    
    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 生成训练批次
        batch = await dataset_generator.generate_training_batch(
            historical_data,
            batch_size=32
        )
        
        # 训练步骤
        loss = await trainer.train_step(batch)
        print(f"Loss: {loss:.4f}")
        
        # 每5个epoch评估一次
        if (epoch + 1) % 5 == 0:
            optimal_strategy = model.find_optimal_strategy()
            if optimal_strategy:
                print("\n最优策略:")
                print(f"球员: {optimal_strategy.player_ids}")
                print(f"阵型: {optimal_strategy.formation}")
                print(f"队长: {optimal_strategy.captain_id}")
                print(f"预期得分: {optimal_strategy.expected_points:.2f}")

if __name__ == "__main__":
    asyncio.run(train_model())