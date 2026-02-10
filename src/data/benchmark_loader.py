"""
Benchmark data loader
Benchmark数据加载器
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .data_types import BenchmarkData, CategoryData, DataPair
from ..utils.image_utils import decode_base64_image


class BenchmarkLoader:
    """
    Benchmark数据集加载器
    
    负责从JSON文件加载benchmark数据，并按类别组织数据
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化数据加载器
        
        Args:
            logger: 日志记录器（可选）
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def load(self, 
             data_path: str, 
             categories: List[str],
             decode_images: bool = False) -> BenchmarkData:
        """
        加载benchmark数据集
        
        Args:
            data_path: JSON文件路径
            categories: 类别名称列表
            decode_images: 是否立即解码base64图像（默认False以节省内存）
            
        Returns:
            BenchmarkData对象
        """
        self.logger.info(f"Loading benchmark data from: {data_path}")
        
        # 读取JSON文件
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.logger.info(f"Raw data loaded, processing...")
        
        # 按类别组织数据
        category_data_dict = {}
        total_pairs = 0
        
        for category in categories:
            pairs = self._extract_category_data(
                raw_data, 
                category, 
                decode_images=decode_images
            )
            
            if pairs:
                category_data_dict[category] = CategoryData(
                    category_name=category,
                    data_pairs=pairs
                )
                total_pairs += len(pairs)
                self.logger.info(f"  - Category '{category}': {len(pairs)} pairs")
            else:
                self.logger.warning(f"  - Category '{category}': No data found")
        
        # 创建BenchmarkData对象
        benchmark_data = BenchmarkData(
            categories=category_data_dict,
            total_pairs=total_pairs,
            category_names=list(category_data_dict.keys()),
            metadata={
                "source_file": str(data_path),
                "categories": categories
            }
        )
        
        self.logger.info(f"Benchmark data loaded successfully: {total_pairs} total pairs")
        
        return benchmark_data
    
    def _extract_category_data(self,
                               raw_data: Dict,
                               category: str,
                               decode_images: bool = False) -> List[DataPair]:
        """
        从原始数据中提取指定类别的数据
        
        Args:
            raw_data: 原始JSON数据（可以是list或dict）
            category: 类别名称
            decode_images: 是否解码图像
            
        Returns:
            DataPair列表
        """
        pairs = []
        
        # 实际数据格式：JSON是一个列表，每个元素包含subset字段
        # 需要从列表中筛选出指定subset的数据
        data_list = []
        
        if isinstance(raw_data, list):
            # 直接是列表格式
            data_list = raw_data
        elif isinstance(raw_data, dict):
            # 如果是字典，尝试获取指定类别
            if category in raw_data:
                category_data = raw_data[category]
                if isinstance(category_data, list):
                    data_list = category_data
        
        # 从列表中筛选指定subset的数据
        for idx, item in enumerate(data_list):
            try:
                # 检查subset是否匹配
                item_subset = item.get("subset", "")
                if item_subset != category:
                    continue  # 跳过不匹配的类别
                
                # 提取必要字段
                # 使用原图路径或索引作为pair_id
                pair_id = item.get("original_image_path", f"{category}_{idx}")
                if pair_id and '/' in pair_id:
                    # 从路径中提取文件名作为ID
                    pair_id = pair_id.split('/')[-1].replace('.png', '').replace('.jpg', '')
                
                # 提取base64图像数据（字段名为 src_img_b64）
                original_image_b64 = item.get("src_img_b64", "")
                
                # 使用英文字段
                edit_instruction = item.get("edit_instruction_en", "")
                original_description = item.get("original_description_en", "")
                rationale = item.get("rationale_short_en", "")  # 新增：加载编辑任务难点说明
                
                # 创建DataPair
                pair = DataPair(
                    pair_id=pair_id,
                    category=category,
                    original_image_b64=original_image_b64,
                    edit_instruction=edit_instruction,
                    original_description=original_description,
                    rationale=rationale,  # 新增：保存rationale
                    metadata=item  # 保存完整的原始数据
                )
                
                # 如果需要，解码图像
                if decode_images and original_image_b64:
                    try:
                        pair.original_image = decode_base64_image(original_image_b64)
                    except Exception as e:
                        self.logger.warning(f"Failed to decode image for {pair_id}: {e}")
                
                pairs.append(pair)
                
            except Exception as e:
                self.logger.error(f"Error processing item {idx} in category {category}: {e}")
                continue
        
        return pairs
    
    def load_from_custom_format(self,
                               data_path: str,
                               category_field: str,
                               id_field: str,
                               image_field: str,
                               instruction_field: str,
                               description_field: str,
                               decode_images: bool = False) -> BenchmarkData:
        """
        从自定义格式的JSON加载数据
        
        Args:
            data_path: JSON文件路径
            category_field: 类别字段名
            id_field: ID字段名
            image_field: 图像base64字段名
            instruction_field: 编辑指令字段名
            description_field: 描述字段名
            decode_images: 是否解码图像
            
        Returns:
            BenchmarkData对象
        """
        self.logger.info(f"Loading benchmark data with custom format from: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 按类别组织
        category_dict = {}
        
        # 假设raw_data是一个列表
        if isinstance(raw_data, list):
            for item in raw_data:
                category = item.get(category_field, "unknown")
                
                pair = DataPair(
                    pair_id=item.get(id_field, ""),
                    category=category,
                    original_image_b64=item.get(image_field, ""),
                    edit_instruction=item.get(instruction_field, ""),
                    original_description=item.get(description_field, ""),
                    metadata=item
                )
                
                if decode_images and pair.original_image_b64:
                    try:
                        pair.original_image = decode_base64_image(pair.original_image_b64)
                    except Exception as e:
                        self.logger.warning(f"Failed to decode image for {pair.pair_id}: {e}")
                
                if category not in category_dict:
                    category_dict[category] = []
                category_dict[category].append(pair)
        
        # 创建CategoryData
        categories = {}
        total_pairs = 0
        for cat_name, pairs in category_dict.items():
            categories[cat_name] = CategoryData(
                category_name=cat_name,
                data_pairs=pairs
            )
            total_pairs += len(pairs)
            self.logger.info(f"  - Category '{cat_name}': {len(pairs)} pairs")
        
        benchmark_data = BenchmarkData(
            categories=categories,
            total_pairs=total_pairs,
            category_names=list(categories.keys()),
            metadata={"source_file": data_path}
        )
        
        self.logger.info(f"Custom format data loaded: {total_pairs} total pairs")
        
        return benchmark_data

