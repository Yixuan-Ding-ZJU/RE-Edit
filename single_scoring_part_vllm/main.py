# -*- coding: utf-8 -*-
"""
主程序入口
支持从YAML配置文件读取参数，命令行参数可以覆盖配置文件中的设置
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from scoring_pipeline import ScoringPipeline


def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[str] = None):
    """
    设置日志
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径（可选）
    """
    handlers = [
        logging.StreamHandler(sys.stdout)  # 输出到终端
    ]
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # 强制重新配置，避免重复配置问题
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    从YAML配置文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_args_with_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """
    合并命令行参数和配置文件参数
    命令行参数优先级更高，会覆盖配置文件中的设置
    
    Args:
        args: 命令行参数
        config: 配置文件字典
        
    Returns:
        合并后的参数对象
    """
    # 从配置文件读取数据配置
    if 'data' in config:
        data_config = config['data']
        if args.json_path is None:
            args.json_path = data_config.get('json_path')
        if args.image_dir is None:
            args.image_dir = data_config.get('image_dir')
        if args.categories is None or len(args.categories) == 0:
            args.categories = data_config.get('categories', ["物理", "环境", "社会", "因果", "指代"])
        if args.image_suffix_pattern is None:
            args.image_suffix_pattern = data_config.get('image_suffix_pattern', '*_primary.png')
    
    # 从配置文件读取模型配置
    if 'model' in config:
        model_config = config['model']
        if args.checkpoint_path is None:
            args.checkpoint_path = model_config.get('checkpoint_path')
        if args.tensor_parallel_size is None:
            tensor_parallel_size = model_config.get('tensor_parallel_size')
            # 处理null值
            if tensor_parallel_size is not None:
                args.tensor_parallel_size = tensor_parallel_size
        if args.gpu_devices is None:
            args.gpu_devices = model_config.get('gpu_devices')
    
    # 从配置文件读取Pipeline配置
    if 'pipeline' in config:
        pipeline_config = config['pipeline']
        if args.batch_size is None:
            args.batch_size = pipeline_config.get('batch_size', 1)
        if args.max_new_tokens is None:
            args.max_new_tokens = pipeline_config.get('max_new_tokens', 512)
        if args.temperature is None:
            args.temperature = pipeline_config.get('temperature', 0)
        if args.log_level is None:
            args.log_level = pipeline_config.get('log_level', 'INFO')
    
    # 从配置文件读取输出配置
    if 'output' in config:
        output_config = config['output']
        if args.output_path is None:
            output_path = output_config.get('output_path')
            if output_path is None:
                # 使用output_dir自动生成
                output_dir = output_config.get('output_dir', './scoring_results')
                args.output_dir = output_dir
            else:
                args.output_path = output_path
    
    # 从配置文件读取默认评分配置
    if 'default_scoring' in config:
        default_scoring = config['default_scoring']
        if args.metric_type is None:
            args.metric_type = default_scoring.get('metric_type')
    
    return args


def main():
    parser = argparse.ArgumentParser(description="批量图像编辑评分系统（基于vLLM + Qwen3-VL）")
    
    # 必需参数（现在可以从配置文件读取，所以设为可选）
    parser.add_argument("--config_path", type=str, required=True,
                       help="YAML配置文件路径（包含所有配置）")
    
    # 可选参数（可以从配置文件读取，也可以通过命令行覆盖）
    parser.add_argument("--json_path", type=str, default=None,
                       help="JSON数据文件路径（覆盖配置文件中的设置）")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="图片目录路径（覆盖配置文件中的设置）")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="模型checkpoint路径（覆盖配置文件中的设置）")
    parser.add_argument("--metric_type", type=str, default=None,
                       choices=["物理", "环境", "社会", "因果", "指代", "pq_metric", "sc_metric", "instruction_following", "reason"],
                       help="评价指标类型（覆盖配置文件中的设置）。'reason'表示按序执行五大类别评价指标")
    
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                       help="类别列表（覆盖配置文件中的设置）")
    parser.add_argument("--output_path", type=str, default=None,
                       help="输出JSON文件路径（覆盖配置文件中的设置）")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                       help="Tensor并行大小（覆盖配置文件中的设置）")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="批处理大小（覆盖配置文件中的设置）")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                       help="最大生成token数（覆盖配置文件中的设置）")
    parser.add_argument("--temperature", type=float, default=None,
                       help="采样温度（覆盖配置文件中的设置）")
    parser.add_argument("--log_level", type=str, default=None,
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别（覆盖配置文件中的设置）")
    parser.add_argument("--image_suffix_pattern", type=str, default=None,
                       help="图片文件名匹配模式（glob模式），默认 '*_primary.png'。例如：'*_primary.png', '*_refined.png', '*.png'")
    parser.add_argument("--gpu_devices", type=str, default=None,
                       help="指定使用的GPU设备，格式如 '0,1,2' 或 '0,2,4,6'。如果指定，会设置CUDA_VISIBLE_DEVICES环境变量")
    
    args = parser.parse_args()
    
    # 加载配置文件
    try:
        config = load_config(args.config_path)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return 1
    
    # 合并配置文件和命令行参数
    args = merge_args_with_config(args, config)
    
    # 验证必需参数
    if args.json_path is None:
        parser.error("--json_path is required (either in config file or command line)")
    if args.image_dir is None:
        parser.error("--image_dir is required (either in config file or command line)")
    if args.checkpoint_path is None:
        parser.error("--checkpoint_path is required (either in config file or command line)")
    if args.metric_type is None:
        parser.error("--metric_type is required (either in config file or command line)")
    
    # 设置默认值
    # 如果metric_type是五大维度之一，且没有指定categories，则只处理该类别
    if args.categories is None or len(args.categories) == 0:
        if args.metric_type in ["物理", "环境", "社会", "因果", "指代"]:
            # 如果metric_type是类别名称，只处理该类别
            args.categories = [args.metric_type]
        elif args.metric_type == "reason":
            # reason模式默认处理所有五大类别
            args.categories = ["物理", "环境", "社会", "因果", "指代"]
        else:
            # 对于pq_metric、sc_metric、instruction_following等，默认处理所有类别
            args.categories = ["物理", "环境", "社会", "因果", "指代"]
    if args.batch_size is None:
        args.batch_size = 1
    if args.max_new_tokens is None:
        args.max_new_tokens = 512
    if args.temperature is None:
        args.temperature = 0
    if args.log_level is None:
        args.log_level = "INFO"
    
    # 生成默认输出路径（先确定输出目录）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_path:
        output_dir = Path(args.output_path).parent
    elif hasattr(args, 'output_dir'):
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.image_dir).parent / "scoring_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output_path is None:
        args.output_path = str(output_dir / f"scoring_results_{args.metric_type}_{timestamp}.json")
    
    # 生成日志文件路径（基于模型、指标和时间）
    # 从checkpoint路径提取模型名称
    checkpoint_name = Path(args.checkpoint_path).name if args.checkpoint_path else "unknown_model"
    # 如果路径很长，尝试提取更短的名称
    if len(checkpoint_name) > 50:
        # 尝试从路径中提取关键部分
        checkpoint_parts = Path(args.checkpoint_path).parts
        if len(checkpoint_parts) > 1:
            checkpoint_name = "_".join(checkpoint_parts[-2:])[:50]
    
    # 清理文件名中的特殊字符
    checkpoint_name = checkpoint_name.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
    metric_name = args.metric_type.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
    
    # 生成日志文件名和路径
    log_filename = f"scoring_log_{checkpoint_name}_{metric_name}_{timestamp}.txt"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_filename
    
    # 设置日志（同时输出到终端和文件）
    setup_logging(args.log_level, str(log_file_path))
    logger = logging.getLogger(__name__)
    
    logger.info(f"Log file: {log_file_path}")
    
    logger.info("=" * 80)
    logger.info("Batch Image Editing Scoring System")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Config file: {args.config_path}")
    logger.info(f"  JSON path: {args.json_path}")
    logger.info(f"  Image dir: {args.image_dir}")
    logger.info(f"  Checkpoint path: {args.checkpoint_path}")
    logger.info(f"  Metric type: {args.metric_type}")
    logger.info(f"  Categories: {args.categories}")
    logger.info(f"  Output path: {args.output_path}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max new tokens: {args.max_new_tokens}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Tensor parallel size: {args.tensor_parallel_size}")
    logger.info(f"  GPU devices: {args.gpu_devices if args.gpu_devices else 'All available GPUs'}")
    logger.info(f"  Log level: {args.log_level}")
    logger.info("=" * 80)
    
    try:
        # 初始化Pipeline
        pipeline = ScoringPipeline(
            checkpoint_path=args.checkpoint_path,
            config_path=args.config_path,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            image_suffix_pattern=args.image_suffix_pattern,
            gpu_devices=args.gpu_devices,
            logger=logger
        )
        
        # 运行评分
        summary = pipeline.run_scoring(
            json_path=args.json_path,
            image_dir=args.image_dir,
            categories=args.categories,
            metric_type=args.metric_type,
            output_path=args.output_path
        )
        
        logger.info("=" * 80)
        logger.info("Scoring completed successfully!")
        logger.info(f"Results saved to: {args.output_path}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in scoring pipeline: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

