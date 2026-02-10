import argparse
import yaml
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import BenchmarkPipeline

from src.iterative_pipeline_v7 import IterativeRefinementPipeline


def load_config(config_path: str) -> dict:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Image Edit Benchmark Evaluation Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "iterative"],
        default="standard",
        help="Pipeline mode: 'standard' for basic evaluation, 'iterative' for iterative refinement (default: standard)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please copy config_template.yaml to config.yaml and edit it.")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # 如果指定了resume，覆盖配置
    if args.resume:
        config.setdefault("evaluation", {})["resume_from_checkpoint"] = True
    
    # 创建并运行pipeline
    try:
        # 根据模式选择不同的Pipeline
        if args.mode == "iterative":
            print("=" * 60)
            print("Running in ITERATIVE REFINEMENT mode")
            print("=" * 60)
            pipeline = IterativeRefinementPipeline(config)
            report = pipeline.run()
        else:
            print("=" * 60)
            print("Running in STANDARD mode")
            print("=" * 60)
            pipeline = BenchmarkPipeline(config)
            report = pipeline.run()
        
        print("\n" + "="*60)
        print("Evaluation Summary:")
        print("="*60)
        
        summary = report.get("summary", {})
        metadata = report.get("metadata", {})
        pipeline_mode = metadata.get("pipeline_mode", "standard")
        
        print(f"Pipeline Mode: {pipeline_mode.upper()}")
        print(f"Total Samples: {summary.get('total_samples', 0)}")
        print(f"Number of Categories: {summary.get('num_categories', 0)}")
        
        # Iterative模式显示更详细的信息
        if pipeline_mode == "iterative_refinement":
            # Yes/No统计
            print(f"\nPrimary Model Yes Rate: {summary.get('primary_overall_mean', 0):.2f}%")
            print(f"Refined Model Yes Rate: {summary.get('refined_overall_mean', 0):.2f}%")
            # 改进/保持/退步/无变化统计
            print(f"Improvement Rate: {summary.get('overall_improvement_rate', 0):.2f}%")
            print(f"Maintained Rate: {summary.get('overall_maintained_rate', 0):.2f}%")
            print(f"Regression Rate: {summary.get('overall_regression_rate', 0):.2f}%")
            print(f"Unchanged Rate: {summary.get('overall_unchanged_rate', 0):.2f}%")
            print(f"Improved Samples: {summary.get('total_improved', 0)}/{summary.get('total_samples', 0)} ({summary.get('improvement_percentage', 0):.1f}%)")
        else:
            print(f"Overall Mean Score: {summary.get('overall_mean', 0):.3f}")
        
        print("\nCategory Yes Rates:")
        for cat, score in summary.get("category_means", {}).items():
            print(f"  - {cat}: {score:.2f}%")
        
        if summary.get("best_category"):
            print(f"\nBest Category: {summary['best_category']['name']} ({summary['best_category']['score']:.2f}%)")
        if summary.get("worst_category"):
            print(f"Worst Category: {summary['worst_category']['name']} ({summary['worst_category']['score']:.2f}%)")
        
        print("="*60)
        print("\n✓ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


