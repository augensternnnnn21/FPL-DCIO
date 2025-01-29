from pathlib import Path
import sys
from dsn_fpl.src.models.fpl_predictor import FPLPredictor

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from dsn_fpl.src.evaluation.model_tester import ModelTester

def main():
    # 设置数据路径
    data_path = project_root / "data"
    
    # 创建你的模型
    model = FPLPredictor()
    
    # 初始化测试器
    tester = ModelTester(data_path, model=model)
    
    # 运行测试
    tester.run_full_test(["2022-23"])
    
    # 生成报告
    tester.generate_report()

if __name__ == "__main__":
    main() 