import os
from pathlib import Path
from config import DEFAULT_CONFIG, COVID_CONFIG
from experiments import ExperimentRunner

def main():
    # 设置基础配置
    base_config = DEFAULT_CONFIG
    base_config.data_path = os.path.join(os.environ['PROJECT_ROOT'], 'data')
    base_config.output_path = os.path.join(os.environ['PROJECT_ROOT'], 'experiments')
    
    # 运行基础实验
    print("\nRunning baseline experiment...")
    runner = ExperimentRunner(base_config)
    runner.run_experiments()
    
    # 运行COVID实验
    print("\nRunning COVID experiment...")
    covid_config = COVID_CONFIG
    covid_config.data_path = base_config.data_path
    covid_config.output_path = os.path.join(base_config.output_path, 'covid')
    
    runner = ExperimentRunner(covid_config)
    runner.run_experiments()

if __name__ == "__main__":
    main()