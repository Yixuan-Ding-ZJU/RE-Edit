# -*- coding: utf-8 -*-
"""
批量评分Pipeline
整合所有模块，实现完整的批量化评分流程
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from vllm import SamplingParams

from data_loader import ScoringDataLoader
from vllm_model import VLLMModel
from prompt_manager import PromptManager
from output_parser import OutputParser
from statistics_calculator import StatisticsCalculator

# 支持的category维度（中文+英文subset，用于metric_type检查）
CATEGORY_METRIC_TYPES = [
    "物理", "环境", "社会", "因果", "指代",
    "physical", "environmental", "cultural", "causal", "referential"
]


class ScoringPipeline:
    """批量评分Pipeline"""
    
    def __init__(self,
                 checkpoint_path: str,
                 config_path: str,
                 tensor_parallel_size: Optional[int] = None,
                 batch_size: int = 1,
                 max_new_tokens: int = 512,
                 temperature: float = 0,
                 image_suffix_pattern: str = "*_primary.png",
                 gpu_devices: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化Pipeline
        
        Args:
            checkpoint_path: 模型checkpoint路径
            config_path: 配置文件路径
            tensor_parallel_size: tensor并行大小
            batch_size: 批处理大小
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            image_suffix_pattern: 图片文件名匹配模式（glob模式），默认 "*_primary.png"
            gpu_devices: 指定使用的GPU设备，格式如 "0,1,2" 或 "0,2,4,6"。如果指定，会设置CUDA_VISIBLE_DEVICES环境变量
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.image_suffix_pattern = image_suffix_pattern
        
        # 初始化各个模块
        self.data_loader = ScoringDataLoader(logger=self.logger)
        self.prompt_manager = PromptManager(config_path=config_path, logger=self.logger)
        self.output_parser = OutputParser(logger=self.logger)
        self.statistics_calculator = StatisticsCalculator(logger=self.logger)
        
        # 初始化vLLM模型
        self.logger.info("Initializing vLLM model...")
        self.vllm_model = VLLMModel(
            checkpoint_path=checkpoint_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_devices=gpu_devices,
            logger=self.logger
        )
        
        # 初始化采样参数
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_k=-1,
            stop_token_ids=[],
        )
        
        self.logger.info("ScoringPipeline initialized")
    
    def cleanup(self):
        """
        清理资源，确保进程完全结束
        应该在任务完成后调用，以便释放GPU内存和其他资源
        """
        self.logger.info("Cleaning up ScoringPipeline resources...")
        
        # 清理vLLM模型
        if hasattr(self, 'vllm_model') and self.vllm_model is not None:
            if hasattr(self.vllm_model, 'llm') and self.vllm_model.llm is not None:
                try:
                    del self.vllm_model.llm
                    self.vllm_model.llm = None
                    self.logger.info("vLLM LLM model deleted")
                except Exception as e:
                    self.logger.warning(f"Error deleting vLLM LLM model: {e}")
            
            if hasattr(self.vllm_model, 'processor') and self.vllm_model.processor is not None:
                try:
                    del self.vllm_model.processor
                    self.vllm_model.processor = None
                    self.logger.info("vLLM processor deleted")
                except Exception as e:
                    self.logger.warning(f"Error deleting vLLM processor: {e}")
            
            del self.vllm_model
            self.vllm_model = None
            self.logger.info("VLLMModel deleted")
        
        # 清理其他模块引用
        if hasattr(self, 'data_loader'):
            del self.data_loader
        if hasattr(self, 'prompt_manager'):
            del self.prompt_manager
        if hasattr(self, 'output_parser'):
            del self.output_parser
        if hasattr(self, 'statistics_calculator'):
            del self.statistics_calculator
        
        self.logger.info("ScoringPipeline cleanup completed")
    
    def prepare_images_for_scoring(self,
                                   data_item: Dict,
                                   use_original_image: bool = True,
                                   use_edited_image: bool = True) -> List[str]:
        """
        准备用于评分的图片列表
        
        Args:
            data_item: 数据项
            use_original_image: 是否使用原始图片
            use_edited_image: 是否使用编辑后的图片
            
        Returns:
            图片列表（base64字符串或URL）
        """
        images = []
        image_id = data_item.get("image_id", "unknown")
        original_status = "not_loaded"
        edited_status = "not_loaded"
        
        if use_original_image:
            # 使用原始图片的base64
            src_img_b64 = data_item.get("src_img_b64", "")
            if src_img_b64:
                # 检查是否是data URI格式
                if src_img_b64.startswith("data:image/"):
                    # 已经是data URI格式，直接使用
                    images.append(src_img_b64)
                    original_status = "loaded (data URI format)"
                else:
                    # 纯base64字符串，转换为data URI格式
                    # 默认使用png格式（因为JSON中的图片通常是png）
                    images.append(f"data:image/png;base64,{src_img_b64}")
                    original_status = "loaded (converted to data URI)"
            else:
                original_status = "missing (src_img_b64 is empty)"
        else:
            original_status = "skipped (use_original_image=False)"
        
        if use_edited_image:
            # 使用编辑后的图片
            primary_image_path = data_item.get("primary_image_path")
            if primary_image_path:
                # 将图片文件转换为base64
                try:
                    img_b64 = self.data_loader.load_image_as_base64(primary_image_path)
                    images.append(img_b64)
                    edited_status = f"loaded from {primary_image_path}"
                except FileNotFoundError as e:
                    edited_status = f"file_not_found: {primary_image_path}"
                    self.logger.error(f"[Image {image_id}] Failed to load edited image: {e}")
                except Exception as e:
                    edited_status = f"error: {str(e)[:50]}"
                    self.logger.error(f"[Image {image_id}] Failed to load edited image {primary_image_path}: {e}")
            else:
                edited_status = "missing (primary_image_path is empty)"
        else:
            edited_status = "skipped (use_edited_image=False)"
        
        # 记录图片准备状态
        self.logger.debug(
            f"[Image {image_id}] Prepared {len(images)} image(s) for scoring: "
            f"original={original_status}, edited={edited_status}"
        )
        
        # 如果图片数量不符合预期，记录警告
        expected_count = sum([use_original_image, use_edited_image])
        if len(images) < expected_count:
            self.logger.warning(
                f"[Image {image_id}] Expected {expected_count} image(s) but got {len(images)}. "
                f"Original: {original_status}, Edited: {edited_status}"
            )
        
        return images
    
    def score_single_item(self,
                         data_item: Dict,
                         metric_type: str,
                         category: Optional[str] = None) -> Dict[str, Any]:
        """
        对单个数据项进行评分
        
        Args:
            data_item: 数据项
            metric_type: 指标类型（类别名称或"pq_metric"/"sc_metric"）
            category: 类别名称（如果metric_type是类别名称，则使用metric_type；否则使用此参数）
            
        Returns:
            评分结果字典
        """
        # 确定实际使用的类别
        if category is None:
            category = metric_type if metric_type in CATEGORY_METRIC_TYPES else data_item.get("subset")
        
        # 获取prompt配置
        if metric_type in CATEGORY_METRIC_TYPES:
            prompt_config = self.prompt_manager.get_category_prompt(metric_type)
        elif metric_type in ["pq_metric", "sc_metric", "instruction_following"]:
            prompt_config = self.prompt_manager.get_metric_prompt(metric_type)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        if not prompt_config:
            raise ValueError(f"No prompt config found for metric: {metric_type}")
        
        system_prompt = prompt_config.get("system_prompt", "")
        user_prompt_template = prompt_config.get("user_prompt_template", "")
        
        # 格式化user prompt
        # 根据metric类型决定需要哪些数据
        if metric_type in CATEGORY_METRIC_TYPES:
            # 五大维度需要所有字段
            user_prompt = self.prompt_manager.format_user_prompt(
                template=user_prompt_template,
                edit_instruction=data_item.get("edit_instruction_en", ""),
                original_description=data_item.get("original_description_en", ""),
                rationale=data_item.get("rationale_short_en", "")
            )
        elif metric_type in ["pq_metric", "sc_metric", "instruction_following"]:
            # PQ、SC和instruction_following指标只需要edit_instruction
            user_prompt = self.prompt_manager.format_user_prompt(
                template=user_prompt_template,
                edit_instruction=data_item.get("edit_instruction_en", ""),
                original_description="",  # 这些指标不需要
                rationale=""  # 这些指标不需要
            )
        else:
            # 默认使用所有字段
            user_prompt = self.prompt_manager.format_user_prompt(
                template=user_prompt_template,
                edit_instruction=data_item.get("edit_instruction_en", ""),
                original_description=data_item.get("original_description_en", ""),
                rationale=data_item.get("rationale_short_en", "")
            )
        
        # 准备图片
        # 所有metric都需要原始图和编辑图
        images = self.prepare_images_for_scoring(
            data_item,
            use_original_image=True,
            use_edited_image=True
        )
        
        # 构建消息
        messages = self.prompt_manager.build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images
        )
        
        # 准备输入
        input_dict = self.vllm_model.prepare_batch_inputs([messages])[0]
        
        # 生成
        outputs = self.vllm_model.generate([input_dict], self.sampling_params)
        
        # 解析输出
        parsed_results = self.output_parser.parse_batch_outputs(outputs)
        
        if parsed_results:
            result = parsed_results[0]
            result["category"] = category
            result["metric_type"] = metric_type
            result["image_id"] = data_item.get("image_id")
            return result
        else:
            return {
                "score": None,
                "reasoning": "",
                "error": "No output generated",
                "category": category,
                "metric_type": metric_type,
                "image_id": data_item.get("image_id")
            }
    
    def score_batch(self,
                    data_items: List[Dict],
                    metric_type: str) -> List[Dict[str, Any]]:
        """
        批量评分
        
        Args:
            data_items: 数据项列表
            metric_type: 指标类型
            
        Returns:
            评分结果列表
        """
        self.logger.info(f"Scoring {len(data_items)} items with metric: {metric_type}")
        
        all_results = []
        
        # 分批处理
        total_batches = (len(data_items) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(data_items), self.batch_size):
            batch = data_items[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            # 准备批量输入
            batch_messages = []
            batch_metadata = []
            
            for data_item in batch:
                category = data_item.get("subset")
                
                # 获取prompt配置
                if metric_type in CATEGORY_METRIC_TYPES:
                    prompt_config = self.prompt_manager.get_category_prompt(metric_type)
                elif metric_type in ["pq_metric", "sc_metric", "instruction_following"]:
                    prompt_config = self.prompt_manager.get_metric_prompt(metric_type)
                else:
                    raise ValueError(f"Unknown metric type: {metric_type}")
                
                if not prompt_config:
                    raise ValueError(f"No prompt config found for metric: {metric_type}")
                
                system_prompt = prompt_config.get("system_prompt", "")
                user_prompt_template = prompt_config.get("user_prompt_template", "")
                
                # 格式化user prompt
                # 根据metric类型决定需要哪些数据
                if metric_type in ["物理", "环境", "社会", "因果", "指代"]:
                    # 五大维度需要所有字段
                    user_prompt = self.prompt_manager.format_user_prompt(
                        template=user_prompt_template,
                        edit_instruction=data_item.get("edit_instruction_en", ""),
                        original_description=data_item.get("original_description_en", ""),
                        rationale=data_item.get("rationale_short_en", "")
                    )
                elif metric_type in ["pq_metric", "sc_metric", "instruction_following"]:
                    # PQ、SC和instruction_following指标只需要edit_instruction
                    user_prompt = self.prompt_manager.format_user_prompt(
                        template=user_prompt_template,
                        edit_instruction=data_item.get("edit_instruction_en", ""),
                        original_description="",  # 这些指标不需要
                        rationale=""  # 这些指标不需要
                    )
                else:
                    # 默认使用所有字段
                    user_prompt = self.prompt_manager.format_user_prompt(
                        template=user_prompt_template,
                        edit_instruction=data_item.get("edit_instruction_en", ""),
                        original_description=data_item.get("original_description_en", ""),
                        rationale=data_item.get("rationale_short_en", "")
                    )
                
                # 准备图片
                # 所有metric都需要原始图和编辑图
                images = self.prepare_images_for_scoring(
                    data_item,
                    use_original_image=True,
                    use_edited_image=True
                )
                
                # 构建消息
                messages = self.prompt_manager.build_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    images=images
                )
                
                batch_messages.append(messages)
                batch_metadata.append({
                    "category": category,
                    "image_id": data_item.get("image_id"),
                    "metric_type": metric_type
                })
            
            # 准备批量输入
            batch_inputs = self.vllm_model.prepare_batch_inputs(batch_messages)
            
            # 批量生成
            try:
                outputs = self.vllm_model.generate(batch_inputs, self.sampling_params)
                
                # 解析输出
                parsed_results = self.output_parser.parse_batch_outputs(outputs)
                
                # 添加元数据
                for j, result in enumerate(parsed_results):
                    result.update(batch_metadata[j])
                    all_results.append(result)
                
                # 实时输出进度
                self.logger.info(f"Batch {batch_num}/{total_batches} completed: {len(parsed_results)} results generated")
                    
            except Exception as e:
                self.logger.error(f"Error in batch generation (batch {batch_num}/{total_batches}): {e}", exc_info=True)
                # 为失败的批次添加错误结果
                for metadata in batch_metadata:
                    all_results.append({
                        "score": None,
                        "reasoning": "",
                        "error": str(e),
                        **metadata
                    })
        
        # 统计成功和失败的数量
        success_count = sum(1 for r in all_results if r.get("score") is not None)
        error_count = len(all_results) - success_count
        self.logger.info(f"Completed scoring: {len(all_results)} total results ({success_count} successful, {error_count} errors)")
        return all_results
    
    def run_scoring(self,
                   json_path: str,
                   image_dir: str,
                   categories: List[str],
                   metric_type: str,
                   output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        运行完整的评分流程
        
        Args:
            json_path: JSON数据文件路径
            image_dir: 图片目录路径
            categories: 类别列表
            metric_type: 指标类型
            output_path: 输出文件路径
            
        Returns:
            完整的统计结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting scoring pipeline")
        self.logger.info(f"JSON path: {json_path}")
        self.logger.info(f"Image dir: {image_dir}")
        self.logger.info(f"Categories: {categories}")
        self.logger.info(f"Metric type: {metric_type}")
        self.logger.info("=" * 60)
        
        # 特殊处理：如果metric_type是"reason"，按序执行五大类别评价
        if metric_type == "reason":
            return self._run_reason_scoring(
                json_path=json_path,
                image_dir=image_dir,
                categories=categories,
                output_path=output_path
            )
        
        # 如果metric_type是五大维度之一，确保只处理该类别
        # 这样可以避免处理不需要的数据
        if metric_type in ["物理", "环境", "社会", "因果", "指代"]:
            if metric_type not in categories:
                self.logger.warning(f"Metric type '{metric_type}' is a category, but it's not in categories list. "
                                  f"Auto-filtering to only process '{metric_type}' category.")
                categories = [metric_type]
            else:
                # 如果metric_type在categories中，只处理该类别（即使categories包含多个）
                self.logger.info(f"Metric type '{metric_type}' is a category. Processing only '{metric_type}' category.")
                categories = [metric_type]
        
        # 1. 加载和匹配数据
        self.logger.info("Step 1: Loading and matching data...")
        data_items = self.data_loader.prepare_data_for_scoring(
            json_path=json_path,
            image_dir=image_dir,
            categories=categories,
            image_suffix_pattern=self.image_suffix_pattern
        )
        
        if not data_items:
            raise ValueError("No data items found after matching")
        
        self.logger.info(f"Loaded {len(data_items)} data items")
        
        # 2. 批量评分
        self.logger.info("Step 2: Running batch scoring...")
        results = self.score_batch(data_items, metric_type)
        
        # 3. 生成统计
        self.logger.info("Step 3: Calculating statistics...")
        
        # 从结果中提取实际评价的类别（避免包含未评价的类别）
        actual_categories_in_results = list(set(r.get("category") for r in results if r.get("category")))
        # 只保留在传入categories列表中的类别
        actual_categories = [cat for cat in actual_categories_in_results if cat in categories]
        
        # 如果没有实际类别，使用传入的categories（可能是全局类别）
        if not actual_categories:
            actual_categories = categories
        
        metadata = {
            "pipeline_mode": "standalone_scoring",
            "scoring_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "data_source": json_path,
            "reward_model": {
                "type": "qwen3_vl_vllm",
                "metric_type": metric_type
            },
            "total_pairs": len(results),
            "categories": actual_categories  # 只包含实际评价的类别
        }
        
        summary = self.statistics_calculator.generate_summary_json(
            results=results,
            categories=actual_categories,  # 只统计实际评价的类别
            metric_name=metric_type,
            metadata=metadata
        )
        
        # 4. 保存结果
        if output_path:
            self.logger.info(f"Step 4: Saving results to {output_path}")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Results saved to {output_path}")
        
        # 输出统计摘要到日志
        self.logger.info("=" * 60)
        self.logger.info("Scoring Summary:")
        self.logger.info(f"  Total samples: {len(results)}")
        self.logger.info(f"  Metric type: {metric_type}")
        if summary.get("overall_statistics"):
            overall = summary["overall_statistics"]
            if "average" in overall:
                self.logger.info(f"  Overall average: {overall['average']:.2f}")
            if "yes_rate" in overall:
                self.logger.info(f"  Overall yes rate: {overall['yes_rate']:.2f}%")
        self.logger.info("=" * 60)
        self.logger.info("Scoring pipeline completed")
        self.logger.info("=" * 60)
        
        return summary
    
    def _run_reason_scoring(self,
                           json_path: str,
                           image_dir: str,
                           categories: List[str],
                           output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        运行reason模式：按序执行五大类别评价指标
        
        Args:
            json_path: JSON数据文件路径
            image_dir: 图片目录路径
            categories: 类别列表（用于过滤数据）
            output_path: 输出文件路径
            
        Returns:
            合并后的统计结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting REASON mode: Sequential evaluation of all 5 categories")
        self.logger.info(f"JSON path: {json_path}")
        self.logger.info(f"Image dir: {image_dir}")
        self.logger.info(f"Categories filter: {categories}")
        self.logger.info("=" * 60)
        
        # 五大类别列表（按顺序）
        reason_categories = ["物理", "环境", "社会", "因果", "指代"]
        
        # 如果指定了categories，只处理这些类别
        if categories:
            reason_categories = [cat for cat in reason_categories if cat in categories]
        
        if not reason_categories:
            raise ValueError("No valid categories found for reason scoring")
        
        self.logger.info(f"Will evaluate categories in order: {reason_categories}")
        
        all_results = []
        category_summaries = {}
        
        # 按序执行每个类别的评价
        for category_metric in reason_categories:
            self.logger.info("=" * 60)
            self.logger.info(f"Evaluating category: {category_metric}")
            self.logger.info("=" * 60)
            
            # 1. 加载和匹配该类别的数据
            self.logger.info(f"Step 1: Loading data for category '{category_metric}'...")
            category_data_items = self.data_loader.prepare_data_for_scoring(
                json_path=json_path,
                image_dir=image_dir,
                categories=[category_metric],  # 只加载当前类别的数据
                image_suffix_pattern=self.image_suffix_pattern
            )
            
            if not category_data_items:
                self.logger.warning(f"No data items found for category '{category_metric}', skipping...")
                continue
            
            self.logger.info(f"Loaded {len(category_data_items)} data items for category '{category_metric}'")
            
            # 2. 使用该类别的metric进行评分（物理类用物理的prompt，环境类用环境的prompt等）
            self.logger.info(f"Step 2: Running scoring for category '{category_metric}'...")
            category_results = self.score_batch(category_data_items, category_metric)
            
            # 3. 生成该类别的统计
            self.logger.info(f"Step 3: Calculating statistics for category '{category_metric}'...")
            
            category_metadata = {
                "pipeline_mode": "reason_mode",
                "scoring_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "data_source": json_path,
                "reward_model": {
                    "type": "qwen3_vl_vllm",
                    "metric_type": category_metric
                },
                "total_pairs": len(category_results),
                "categories": [category_metric]
            }
            
            category_summary = self.statistics_calculator.generate_summary_json(
                results=category_results,
                categories=[category_metric],
                metric_name=category_metric,
                metadata=category_metadata
            )
            
            category_summaries[category_metric] = category_summary
            all_results.extend(category_results)
            
            # 输出该类别的统计摘要
            self.logger.info(f"Category '{category_metric}' Summary:")
            if category_summary.get("overall_statistics"):
                overall = category_summary["overall_statistics"]
                if "average" in overall:
                    self.logger.info(f"  Average: {overall['average']:.2f}")
                if "yes_rate" in overall:
                    self.logger.info(f"  Yes rate: {overall['yes_rate']:.2f}%")
            self.logger.info("=" * 60)
        
        # 4. 生成合并后的统计报告
        self.logger.info("Step 4: Generating combined statistics...")
        
        # 从结果中提取实际评价的类别
        actual_categories = list(set(r.get("category") for r in all_results if r.get("category")))
        
        combined_metadata = {
            "pipeline_mode": "reason_mode_combined",
            "scoring_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "data_source": json_path,
            "reward_model": {
                "type": "qwen3_vl_vllm",
                "metric_type": "reason",
                "evaluated_categories": actual_categories
            },
            "total_pairs": len(all_results),
            "categories": actual_categories
        }
        
        # 生成合并统计（使用第一个类别的metric_name，但实际上每个类别都有自己的metric）
        combined_summary = self.statistics_calculator.generate_summary_json(
            results=all_results,
            categories=actual_categories,
            metric_name="reason",  # 使用"reason"作为合并后的metric名称
            metadata=combined_metadata
        )
        
        # 添加每个类别的详细统计
        combined_summary["category_details"] = category_summaries
        
        # 5. 保存结果
        if output_path:
            self.logger.info(f"Step 5: Saving results to {output_path}")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Results saved to {output_path}")
        
        # 输出最终统计摘要
        self.logger.info("=" * 60)
        self.logger.info("REASON Mode - Final Summary:")
        self.logger.info(f"  Total samples: {len(all_results)}")
        self.logger.info(f"  Evaluated categories: {actual_categories}")
        if combined_summary.get("overall_statistics"):
            overall = combined_summary["overall_statistics"]
            if "average" in overall:
                self.logger.info(f"  Overall average: {overall['average']:.2f}")
            if "yes_rate" in overall:
                self.logger.info(f"  Overall yes rate: {overall['yes_rate']:.2f}%")
        
        # 输出每个类别的平均分
        if combined_summary.get("category_statistics"):
            self.logger.info("  Category averages:")
            for cat, stats in combined_summary["category_statistics"].items():
                avg = stats.get("average", 0)
                self.logger.info(f"    {cat}: {avg:.2f}")
        
        self.logger.info("=" * 60)
        self.logger.info("REASON mode scoring completed")
        self.logger.info("=" * 60)
        
        return combined_summary

