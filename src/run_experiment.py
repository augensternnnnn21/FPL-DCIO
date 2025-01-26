import os
import torch
from pathlib import Path
from config import ExperimentConfig, get_default_config
from experiments import ExperimentRunner

def main():
    print("\nDEBUG: Starting main...")
    
    # 使用正确的配置初始化方式
    config = get_default_config()  # 替换原有的ExperimentConfig()
    
    # 创建实验运行器
    try:
        runner = ExperimentRunner(config)
        runner.run_experiments()
    except Exception as e:
        print(f"Error running experiments: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()