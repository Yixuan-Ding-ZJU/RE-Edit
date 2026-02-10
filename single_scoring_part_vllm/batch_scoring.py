# -*- coding: utf-8 -*-
"""
批量评分任务执行脚本
确保每个任务完全结束后（进程全部结束）才执行下一个任务
"""

import logging
import sys
import json
import time
import gc
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from scoring_pipeline import ScoringPipeline


@dataclass
class TaskConfig:
    """任务配置"""
    name: str  # 任务名称
    json_path: str  # JSON数据文件路径
    image_dir: str  # 图片目录路径
    metric_type: str  # 评价指标类型
    output_path: Optional[str] = None  # 输出文件路径（如果为None则自动生成）
    categories: Optional[List[str]] = None  # 类别列表（如果为None则使用默认值）
    checkpoint_path: Optional[str] = None  # 模型路径（如果为None则使用配置文件中的）
    config_path: str = "scoring_config.yaml"  # 配置文件路径
    tensor_parallel_size: Optional[int] = None  # Tensor并行大小
    batch_size: Optional[int] = None  # 批处理大小
    max_new_tokens: Optional[int] = None  # 最大生成token数
    temperature: Optional[float] = None  # 采样温度
    image_suffix_pattern: str = "*_primary.png"  # 图片文件名匹配模式（glob模式）
    gpu_devices: Optional[str] = None  # 指定使用的GPU设备，格式如 "0,1,2"
    log_level: str = "INFO"  # 日志级别


@dataclass
class TaskResult:
    """任务执行结果"""
    name: str
    status: str  # "pending", "running", "completed", "failed", "skipped"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[float] = None  # 执行时长（秒）
    output_path: Optional[str] = None
    error_message: Optional[str] = None


class BatchScoringRunner:
    """批量评分执行器"""
    
    def __init__(self,
                 tasks: List[TaskConfig],
                 results_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化批量执行器
        
        Args:
            tasks: 任务列表
            results_dir: 结果保存目录（如果为None则使用当前目录下的batch_results）
            logger: 日志记录器
        """
        self.tasks = tasks
        self.logger = logger or logging.getLogger(__name__)
        
        # 设置结果目录
        if results_dir is None:
            results_dir = Path.cwd() / "batch_results"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 任务结果列表
        self.task_results: List[TaskResult] = []
        
        # 执行状态文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.status_file = self.results_dir / f"batch_status_{timestamp}.json"
        self.execution_log = self.results_dir / f"batch_execution_log_{timestamp}.txt"
        
        # 设置执行日志
        self._setup_execution_logger()
    
    def _setup_execution_logger(self):
        """设置执行日志"""
        file_handler = logging.FileHandler(self.execution_log, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _cleanup_resources(self, pipeline: Optional[ScoringPipeline] = None):
        """
        清理资源，确保进程完全结束
        
        Args:
            pipeline: ScoringPipeline实例（如果提供则清理其资源）
        """
        self.logger.info("Cleaning up resources...")
        
        if pipeline is not None:
            # 使用Pipeline的cleanup方法
            try:
                pipeline.cleanup()
            except Exception as e:
                self.logger.warning(f"Error during pipeline cleanup: {e}")
            
            # 删除pipeline引用
            del pipeline
            self.logger.info("ScoringPipeline reference deleted")
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared")
        
        # 强制垃圾回收
        gc.collect()
        self.logger.info("Garbage collection completed")
        
        # 等待一小段时间确保资源完全释放
        time.sleep(2)
        self.logger.info("Resource cleanup completed")
    
    def _save_status(self):
        """保存执行状态"""
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(self.tasks),
            "completed_tasks": sum(1 for r in self.task_results if r.status == "completed"),
            "failed_tasks": sum(1 for r in self.task_results if r.status == "failed"),
            "running_tasks": sum(1 for r in self.task_results if r.status == "running"),
            "pending_tasks": sum(1 for r in self.task_results if r.status == "pending"),
            "task_results": [asdict(r) for r in self.task_results]
        }
        
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)
    
    def _execute_single_task(self, task: TaskConfig) -> TaskResult:
        """
        执行单个任务
        
        Args:
            task: 任务配置
            
        Returns:
            任务执行结果
        """
        result = TaskResult(name=task.name, status="running")
        result.start_time = datetime.now().isoformat()
        
        pipeline = None
        
        try:
            self.logger.info("=" * 80)
            self.logger.info(f"Starting task: {task.name}")
            self.logger.info(f"  JSON path: {task.json_path}")
            self.logger.info(f"  Image dir: {task.image_dir}")
            self.logger.info(f"  Metric type: {task.metric_type}")
            self.logger.info(f"  Output path: {task.output_path}")
            self.logger.info("=" * 80)
            
            # 验证文件路径
            if not Path(task.json_path).exists():
                raise FileNotFoundError(f"JSON file not found: {task.json_path}")
            if not Path(task.image_dir).exists():
                raise FileNotFoundError(f"Image directory not found: {task.image_dir}")
            if not Path(task.config_path).exists():
                raise FileNotFoundError(f"Config file not found: {task.config_path}")
            
            # 确定输出路径
            output_path = task.output_path
            if output_path is None:
                # 自动生成输出路径
                output_dir = Path(task.image_dir) / "batch_scoring_results"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metric_name = task.metric_type.replace("/", "_").replace("\\", "_").replace(" ", "_")
                task_name = task.name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                output_path = str(output_dir / f"scoring_results_{task_name}_{metric_name}_{timestamp}.json")
            
            # 确定类别
            categories = task.categories
            if categories is None or len(categories) == 0:
                if task.metric_type in ["物理", "环境", "社会", "因果", "指代"]:
                    categories = [task.metric_type]
                elif task.metric_type == "reason":
                    categories = ["物理", "环境", "社会", "因果", "指代"]
                else:
                    categories = ["物理", "环境", "社会", "因果", "指代"]
            
            # 确定checkpoint路径（从配置文件或任务配置中获取）
            checkpoint_path = task.checkpoint_path
            if checkpoint_path is None:
                # 从配置文件读取
                import yaml
                with open(task.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                checkpoint_path = config.get('model', {}).get('checkpoint_path')
                if checkpoint_path is None:
                    raise ValueError("checkpoint_path not found in config file and not provided in task")
            
            # 初始化Pipeline
            self.logger.info(f"Initializing ScoringPipeline for task: {task.name}")
            pipeline = ScoringPipeline(
                checkpoint_path=checkpoint_path,
                config_path=task.config_path,
                tensor_parallel_size=task.tensor_parallel_size,
                batch_size=task.batch_size or 1,
                max_new_tokens=task.max_new_tokens or 512,
                temperature=task.temperature or 0,
                image_suffix_pattern=task.image_suffix_pattern,
                gpu_devices=task.gpu_devices,
                logger=self.logger
            )
            
            # 执行评分
            self.logger.info(f"Running scoring for task: {task.name}")
            summary = pipeline.run_scoring(
                json_path=task.json_path,
                image_dir=task.image_dir,
                categories=categories,
                metric_type=task.metric_type,
                output_path=output_path
            )
            
            # 记录成功
            result.status = "completed"
            result.output_path = output_path
            result.end_time = datetime.now().isoformat()
            
            start_dt = datetime.fromisoformat(result.start_time)
            end_dt = datetime.fromisoformat(result.end_time)
            result.duration = (end_dt - start_dt).total_seconds()
            
            self.logger.info("=" * 80)
            self.logger.info(f"Task '{task.name}' completed successfully!")
            self.logger.info(f"  Duration: {result.duration:.2f} seconds")
            self.logger.info(f"  Output: {output_path}")
            self.logger.info("=" * 80)
            
        except Exception as e:
            # 记录失败
            result.status = "failed"
            result.error_message = str(e)
            result.end_time = datetime.now().isoformat()
            
            if result.start_time:
                start_dt = datetime.fromisoformat(result.start_time)
                end_dt = datetime.fromisoformat(result.end_time)
                result.duration = (end_dt - start_dt).total_seconds()
            
            self.logger.error("=" * 80)
            self.logger.error(f"Task '{task.name}' failed!")
            self.logger.error(f"  Error: {result.error_message}")
            self.logger.error(f"  Duration: {result.duration:.2f} seconds" if result.duration else "  Duration: N/A")
            self.logger.error("=" * 80, exc_info=True)
        
        finally:
            # 清理资源
            self._cleanup_resources(pipeline)
            pipeline = None
        
        return result
    
    def run(self, skip_completed: bool = False):
        """
        执行所有任务
        
        Args:
            skip_completed: 是否跳过已完成的任务（从状态文件中读取）
        """
        self.logger.info("=" * 80)
        self.logger.info("Batch Scoring Runner")
        self.logger.info("=" * 80)
        self.logger.info(f"Total tasks: {len(self.tasks)}")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Status file: {self.status_file}")
        self.logger.info(f"Execution log: {self.execution_log}")
        self.logger.info("=" * 80)
        
        # 初始化任务结果
        if skip_completed and self.status_file.exists():
            # 从状态文件加载已完成的任务
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                    existing_results = {
                        r['name']: TaskResult(**r) 
                        for r in status_data.get('task_results', [])
                    }
                    
                    # 只保留已完成的任务
                    for task in self.tasks:
                        if task.name in existing_results:
                            existing_result = existing_results[task.name]
                            if existing_result.status == "completed":
                                self.task_results.append(existing_result)
                                self.logger.info(f"Skipping completed task: {task.name}")
                            else:
                                # 重新执行失败或未完成的任务
                                self.task_results.append(TaskResult(name=task.name, status="pending"))
                        else:
                            self.task_results.append(TaskResult(name=task.name, status="pending"))
            except Exception as e:
                self.logger.warning(f"Failed to load status file: {e}, starting fresh")
                self.task_results = [TaskResult(name=task.name, status="pending") for task in self.tasks]
        else:
            self.task_results = [TaskResult(name=task.name, status="pending") for task in self.tasks]
        
        # 执行任务
        total_start_time = time.time()
        
        for i, task in enumerate(self.tasks, 1):
            # 检查是否已完成
            existing_result = next((r for r in self.task_results if r.name == task.name), None)
            if skip_completed and existing_result and existing_result.status == "completed":
                self.logger.info(f"[{i}/{len(self.tasks)}] Skipping completed task: {task.name}")
                continue
            
            self.logger.info(f"[{i}/{len(self.tasks)}] Processing task: {task.name}")
            
            # 更新状态为running
            result = next((r for r in self.task_results if r.name == task.name), None)
            if result:
                result.status = "running"
            else:
                result = TaskResult(name=task.name, status="running")
                self.task_results.append(result)
            
            # 执行任务
            result = self._execute_single_task(task)
            
            # 更新结果
            for j, r in enumerate(self.task_results):
                if r.name == task.name:
                    self.task_results[j] = result
                    break
            else:
                self.task_results.append(result)
            
            # 保存状态
            self._save_status()
            
            # 任务之间的间隔（确保资源完全释放）
            if i < len(self.tasks):
                self.logger.info("Waiting before next task...")
                time.sleep(3)
        
        # 生成最终报告
        total_duration = time.time() - total_start_time
        self._generate_summary(total_duration)
    
    def _generate_summary(self, total_duration: float):
        """生成执行摘要"""
        self.logger.info("=" * 80)
        self.logger.info("Batch Execution Summary")
        self.logger.info("=" * 80)
        self.logger.info(f"Total tasks: {len(self.tasks)}")
        self.logger.info(f"Completed: {sum(1 for r in self.task_results if r.status == 'completed')}")
        self.logger.info(f"Failed: {sum(1 for r in self.task_results if r.status == 'failed')}")
        self.logger.info(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        self.logger.info("=" * 80)
        
        # 详细任务列表
        self.logger.info("\nTask Details:")
        for result in self.task_results:
            status_icon = "✓" if result.status == "completed" else "✗" if result.status == "failed" else "○"
            self.logger.info(f"  {status_icon} {result.name}: {result.status}")
            if result.duration:
                self.logger.info(f"    Duration: {result.duration:.2f} seconds")
            if result.output_path:
                self.logger.info(f"    Output: {result.output_path}")
            if result.error_message:
                self.logger.info(f"    Error: {result.error_message}")
        
        self.logger.info("=" * 80)
        self.logger.info(f"Status file saved to: {self.status_file}")
        self.logger.info(f"Execution log saved to: {self.execution_log}")
        self.logger.info("=" * 80)


def create_example_tasks() -> List[TaskConfig]:
    """
    创建示例任务列表
    可以根据实际需求修改
    """
    base_config_path = "scoring_config.yaml"
    base_json_path = "/mnt/autodl_tmp1/dyx/Benchmark/version_4_simplified_v2_translated_with_images.json"
    base_image_dir = "/mnt/autodl_tmp1/dyx/image_edit_benchmark_12_14_grab_from_h100/results_iterative/ICML_12_18_flux_2"
    
    tasks = [
        # 示例1: 执行五大维度评价
        TaskConfig(
            name="flux2_物理",
            json_path=base_json_path,
            image_dir=base_image_dir,
            metric_type="物理",
            config_path=base_config_path,
            batch_size=6,
            log_level="INFO"
        ),
        TaskConfig(
            name="flux2_环境",
            json_path=base_json_path,
            image_dir=base_image_dir,
            metric_type="环境",
            config_path=base_config_path,
            batch_size=6,
            log_level="INFO"
        ),
        TaskConfig(
            name="flux2_社会",
            json_path=base_json_path,
            image_dir=base_image_dir,
            metric_type="社会",
            config_path=base_config_path,
            batch_size=6,
            log_level="INFO"
        ),
        TaskConfig(
            name="flux2_因果",
            json_path=base_json_path,
            image_dir=base_image_dir,
            metric_type="因果",
            config_path=base_config_path,
            batch_size=6,
            log_level="INFO"
        ),
        TaskConfig(
            name="flux2_指代",
            json_path=base_json_path,
            image_dir=base_image_dir,
            metric_type="指代",
            config_path=base_config_path,
            batch_size=6,
            log_level="INFO"
        ),
        
        # 示例2: 执行其他指标
        TaskConfig(
            name="flux2_instruction_following",
            json_path=base_json_path,
            image_dir=base_image_dir,
            metric_type="instruction_following",
            config_path=base_config_path,
            batch_size=6,
            log_level="INFO"
        ),
        
        # 示例3: 执行reason模式
        # TaskConfig(
        #     name="flux2_reason",
        #     json_path=base_json_path,
        #     image_dir=base_image_dir,
        #     metric_type="reason",
        #     config_path=base_config_path,
        #     batch_size=6,
        #     log_level="INFO"
        # ),
    ]
    
    return tasks


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="批量评分任务执行脚本")
    parser.add_argument("--tasks", type=str, default=None,
                       help="任务配置文件路径（JSON格式），如果未指定则使用示例任务")
    parser.add_argument("--results_dir", type=str, default=None,
                       help="结果保存目录（默认：当前目录下的batch_results）")
    parser.add_argument("--skip_completed", action="store_true",
                       help="跳过已完成的任务（从状态文件中读取）")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 加载任务
    if args.tasks:
        # 从JSON文件加载任务
        with open(args.tasks, 'r', encoding='utf-8') as f:
            tasks_data = json.load(f)
            tasks = [TaskConfig(**task_dict) for task_dict in tasks_data.get('tasks', [])]
    else:
        # 使用示例任务
        tasks = create_example_tasks()
        logger.info("Using example tasks. To use custom tasks, create a JSON file and use --tasks option.")
    
    if not tasks:
        logger.error("No tasks to execute!")
        return 1
    
    # 创建执行器并运行
    runner = BatchScoringRunner(
        tasks=tasks,
        results_dir=args.results_dir,
        logger=logger
    )
    
    try:
        runner.run(skip_completed=args.skip_completed)
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        runner._save_status()
        return 1
    except Exception as e:
        logger.error(f"Error in batch execution: {e}", exc_info=True)
        runner._save_status()
        return 1


if __name__ == "__main__":
    sys.exit(main())

