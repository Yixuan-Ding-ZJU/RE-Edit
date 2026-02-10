# -*- coding: utf-8 -*-
"""
数据加载模块
从JSON文件加载benchmark数据，并匹配对应的图片文件
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging
import base64


class ScoringDataLoader:
    """评分数据加载器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def load_json_data(self, json_path: str, categories: List[str]) -> List[Dict]:
        """
        从JSON文件加载数据
        
        Args:
            json_path: JSON文件路径
            categories: 类别列表
            
        Returns:
            数据列表，每个元素包含必要字段
        """
        self.logger.info(f"Loading JSON data from: {json_path}")
        
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 处理数据格式（可能是list或dict）
        data_list = []
        if isinstance(raw_data, list):
            data_list = raw_data
        elif isinstance(raw_data, dict):
            # 如果是字典，尝试获取所有类别的数据
            for category in categories:
                if category in raw_data:
                    cat_data = raw_data[category]
                    if isinstance(cat_data, list):
                        data_list.extend(cat_data)
        
        # 提取必要字段
        extracted_data = []
        for idx, item in enumerate(data_list):
            try:
                # 只处理指定类别的数据
                item_subset = item.get("subset", "")
                if item_subset not in categories:
                    continue
                
                # 提取字段
                data_item = {
                    "subset": item_subset,
                    "edit_instruction_en": item.get("edit_instruction_en", ""),
                    "rationale_short_en": item.get("rationale_short_en", ""),
                    "src_img_b64": item.get("src_img_b64", ""),
                    "original_image_path": item.get("original_image_path", ""),
                    "original_description_en": item.get("original_description_en", ""),
                    "raw_data": item  # 保存完整原始数据
                }
                
                # 从original_image_path中提取编号
                image_id = self._extract_image_id(data_item["original_image_path"])
                data_item["image_id"] = image_id
                
                extracted_data.append(data_item)
                
            except Exception as e:
                self.logger.warning(f"Error processing item {idx}: {e}")
                continue
        
        self.logger.info(f"Loaded {len(extracted_data)} data items from JSON")
        return extracted_data
    
    def _extract_image_id(self, image_path: str) -> Optional[str]:
        """
        从original_image_path中提取图片编号
        
        Args:
            image_path: 图片路径，如 "images/00025.png" 或 "00025.png"
            
        Returns:
            提取的编号，如 "00025"
        """
        if not image_path:
            return None
        
        # 提取文件名（去除路径）
        filename = Path(image_path).stem  # 去除扩展名
        
        # 提取数字部分
        match = re.search(r'(\d+)', filename)
        if match:
            return match.group(1)
        
        return None
    
    def match_primary_images(self, 
                           data_items: List[Dict], 
                           image_dir: str,
                           categories: List[str],
                           image_suffix_pattern: str = "*_primary.png") -> List[Dict]:
        """
        匹配编辑后的图片文件
        
        Args:
            data_items: JSON数据列表
            image_dir: 图片目录路径
            categories: 类别列表
            image_suffix_pattern: 图片文件名匹配模式（glob模式），默认 "*_primary.png"
                                 例如："*_primary.png", "*_refined.png", "*.png" 等
            
        Returns:
            添加了primary_image_path字段的数据列表
        """
        self.logger.info(f"Matching images from: {image_dir}")
        self.logger.info(f"Using image suffix pattern: {image_suffix_pattern}")
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # 构建图片路径映射：{category: {image_id: image_path}}
        image_map = {}
        for category in categories:
            category_dir = image_dir / category
            if not category_dir.exists():
                self.logger.warning(f"Category directory not found: {category_dir}")
                continue
            
            image_map[category] = {}
            
            # 扫描该类别目录下匹配模式的图片
            for img_file in category_dir.glob(image_suffix_pattern):
                # 从文件名提取编号，格式：{编号}_{类别}_{难度}_xxx.png
                # 支持多种格式：00025_物理_medium_primary.png, 00025.png 等
                filename = img_file.stem  # 去除扩展名
                match = re.match(r'(\d+)_', filename)
                if match:
                    image_id = match.group(1)
                    image_map[category][image_id] = str(img_file)
                else:
                    # 如果文件名直接是数字，也尝试提取
                    match = re.match(r'^(\d+)$', filename)
                    if match:
                        image_id = match.group(1)
                        image_map[category][image_id] = str(img_file)
            
            self.logger.info(f"Found {len(image_map[category])} images matching pattern '{image_suffix_pattern}' in {category}")
        
        # 匹配数据
        matched_data = []
        for item in data_items:
            category = item.get("subset")
            image_id = item.get("image_id")
            
            if category and image_id and category in image_map:
                if image_id in image_map[category]:
                    item["primary_image_path"] = image_map[category][image_id]
                    matched_data.append(item)
                else:
                    self.logger.warning(f"No primary image found for {category}/{image_id}")
            else:
                self.logger.warning(f"Missing category or image_id for item: {item.get('original_image_path')}")
        
        self.logger.info(f"Matched {len(matched_data)} data items with primary images")
        return matched_data
    
    def load_image_as_base64(self, image_path: str) -> str:
        """
        将图片文件加载为base64编码（data URI格式）
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            data URI格式的base64编码字符串（用于vLLM）
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # 转换为base64
        base64_str = base64.b64encode(image_data).decode('utf-8')
        
        # 根据文件扩展名确定图片格式
        image_format = image_path.suffix.lower().lstrip('.')
        if image_format not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
            image_format = 'png'  # 默认使用png
        
        # 返回data URI格式，这样process_vision_info可以正确识别
        return f"data:image/{image_format};base64,{base64_str}"
    
    def convert_base64_to_data_uri(self, base64_str: str, image_format: str = "png") -> str:
        """
        将base64字符串转换为data URI格式
        
        Args:
            base64_str: base64编码的字符串
            image_format: 图片格式（png, jpeg等）
            
        Returns:
            data URI格式的字符串
        """
        return f"data:image/{image_format};base64,{base64_str}"
    
    def prepare_data_for_scoring(self,
                                json_path: str,
                                image_dir: str,
                                categories: List[str],
                                image_suffix_pattern: str = "*_primary.png") -> List[Dict]:
        """
        完整的数据准备流程
        
        Args:
            json_path: JSON文件路径
            image_dir: 图片目录路径
            categories: 类别列表
            image_suffix_pattern: 图片文件名匹配模式（glob模式），默认 "*_primary.png"
                                 例如："*_primary.png", "*_refined.png", "*.png" 等
            
        Returns:
            准备好的数据列表，包含所有必要字段
        """
        # 1. 加载JSON数据
        data_items = self.load_json_data(json_path, categories)
        
        # 2. 匹配编辑后的图片
        matched_data = self.match_primary_images(
            data_items, 
            image_dir, 
            categories,
            image_suffix_pattern=image_suffix_pattern
        )
        
        return matched_data

