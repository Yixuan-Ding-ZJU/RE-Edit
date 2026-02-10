"""
Base MLLM Model
MLLM基类 - 定义统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from PIL import Image
import logging

from ..base import BaseModel


class BaseMllm(BaseModel):
    """
    MLLM (Multimodal Large Language Model) 基类
    
    用于分析图像编辑结果，生成改进建议和Re-edit指令
    """
    
    def __init__(self, config: dict):
        """
        初始化MLLM模型
        
        Args:
            config: 配置字典，应包含：
                - model_name: 模型路径或名称
                - device: 设备配置 (e.g., "cuda:0", "auto")
                - dtype: 数据类型 (e.g., "bfloat16", "float16")
                - batch_size: 批处理大小
                - max_new_tokens: 最大生成token数
        """
        # 先设置logger，因为_initialize可能会用到
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(config)
    
    @abstractmethod
    def analyze(self,
                edited_image: Image.Image,
                original_image: Image.Image,
                original_description: str,
                edit_instruction: str,
                category: str,
                **kwargs) -> Dict[str, str]:
        """
        分析单个图像编辑结果，生成改进建议
        
        Args:
            edited_image: 编辑后的图像
            original_image: 原始图像
            original_description: 原图描述
            edit_instruction: 编辑指令
            category: 类别名称（用于选择合适的prompt）
            **kwargs: 其他参数
        
        Returns:
            字典，包含：
            {
                "cot": "Chain of Thought推理过程...",
                "re_edit_instruction": "改进编辑指令...",
                "raw_output": "模型原始输出（可选）"
            }
        """
        pass
    
    @abstractmethod
    def batch_analyze(self,
                      edited_images: List[Image.Image],
                      original_images: List[Image.Image],
                      original_descriptions: List[str],
                      edit_instructions: List[str],
                      categories: List[str],
                      batch_size: int = 4,
                      **kwargs) -> List[Dict[str, str]]:
        """
        批量分析图像编辑结果
        
        Args:
            edited_images: 编辑后的图像列表
            original_images: 原始图像列表
            original_descriptions: 原图描述列表
            edit_instructions: 编辑指令列表
            categories: 类别名称列表
            batch_size: 批处理大小
            **kwargs: 其他参数
        
        Returns:
            结果列表，每个元素为：
            {
                "cot": "...",
                "re_edit_instruction": "...",
                "raw_output": "..."
            }
        """
        pass
    
    def parse_output(self, raw_output: str) -> Dict[str, str]:
        """
        解析MLLM输出，提取CoT和Re-edit指令
        
        支持多种格式：
        1. JSON格式: {"cot": "...", "re_edit_instruction": "..."}
        2. Markdown格式: ## CoT \n ... \n ## Re-edit Instruction \n ...
        3. 纯文本格式: 使用特定标记分隔
        
        Args:
            raw_output: MLLM原始输出
        
        Returns:
            解析后的字典
        """
        # TODO: 具体实现由子类完成或在此提供默认实现
        # 默认实现：假设整个输出就是re_edit_instruction
        return {
            "cot": "",
            "re_edit_instruction": raw_output.strip(),
            "raw_output": raw_output
        }
    
    def validate_output(self, output: Dict[str, str]) -> bool:
        """
        验证MLLM输出是否有效
        
        Args:
            output: 解析后的输出字典
        
        Returns:
            True如果输出有效，否则False
        """
        # 至少要有re_edit_instruction且非空
        re_edit = output.get("re_edit_instruction", "").strip()
        return len(re_edit) > 0
    
    def get_fallback_instruction(self, 
                                  original_instruction: str,
                                  category: str) -> str:
        """
        当MLLM输出无效时，返回fallback指令
        
        Args:
            original_instruction: 原始编辑指令
            category: 类别名称
        
        Returns:
            fallback指令（通常是原始指令的轻微变体）
        """
        return f"{original_instruction} (with enhanced details)"
    
    def load_to_gpu(self):
        """
        将模型加载到GPU
        
        子类应实现具体的GPU加载逻辑
        """
        self.logger.info(f"[{self.__class__.__name__}] Loading model to GPU...")
        # 默认实现：调用_initialize
        if not hasattr(self, 'model') or self.model is None:
            self._initialize()
    
    def unload_from_gpu(self):
        """
        从GPU卸载模型
        
        子类应实现具体的GPU卸载逻辑
        """
        self.logger.info(f"[{self.__class__.__name__}] Unloading model from GPU...")
        # 默认实现：清理model和相关资源
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

