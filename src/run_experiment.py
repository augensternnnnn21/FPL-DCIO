import os
import sys
import torch
import numpy as np
from experiments import ExperimentRunner
from config import get_default_config, COVID_CONFIG

def main():
    print("\nDEBUG: Starting main...")
    config = get_default_config()
    
    print("Config type:", type(config))
    print("Config dir:", dir(config))
    print("Config dict:", config.__dict__ if hasattr(config, '__dict__') else "No __dict__")
    print("Config data_path:", getattr(config, 'data_path', 'Not found'))
    
    # 设置随机种子
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # 创建实验目录
    os.makedirs("experiments", exist_ok=True)
    
    print("\nRunning default experiment configuration...")
    # 使用默认配置运行实验
    runner = ExperimentRunner(config)
    runner.run_experiments()
    
    print("\nRunning COVID experiment configuration...")
    # 使用COVID配置运行实验
    covid_runner = ExperimentRunner(COVID_CONFIG)
    covid_runner.run_experiments()

if __name__ == "__main__":
    main() 