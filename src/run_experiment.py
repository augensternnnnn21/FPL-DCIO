import os
import torch
import numpy as np
from experiments import ExperimentRunner
from config.experiment_config import DEFAULT_CONFIG, COVID_CONFIG

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建实验目录
    os.makedirs("experiments", exist_ok=True)
    
    print("Running default experiment configuration...")
    # 使用默认配置运行实验
    runner = ExperimentRunner(DEFAULT_CONFIG)
    runner.run_experiments()
    
    print("\nRunning COVID experiment configuration...")
    # 使用COVID配置运行实验
    covid_runner = ExperimentRunner(COVID_CONFIG)
    covid_runner.run_experiments()

if __name__ == "__main__":
    main() 