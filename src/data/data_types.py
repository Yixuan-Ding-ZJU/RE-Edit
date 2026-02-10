"""
Data type definitions
数据类型定义
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from PIL import Image


@dataclass
class DataPair:
    """
    单个数据对
    
    Attributes:
        pair_id: 数据对的唯一标识
        category: 类别名称
        original_image_b64: 原始图像的base64编码
        edit_instruction: 编辑指令
        original_description: 原始图像描述
        rationale: 编辑任务的难点说明（可选）
        original_image: 解码后的PIL图像对象（可选）
        edited_image: 编辑后的图像（可选）
        score: 评分（可选）
        metadata: 其他元数据
    """
    pair_id: str
    category: str
    original_image_b64: str
    edit_instruction: str
    original_description: str
    rationale: Optional[str] = None  # 新增：编辑任务难点说明
    original_image: Optional[Image.Image] = None
    edited_image: Optional[Image.Image] = None
    score: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class CategoryData:
    """
    单个类别的数据集合
    
    Attributes:
        category_name: 类别名称
        data_pairs: 该类别的所有数据对列表
        scores: 该类别所有数据对的评分列表
        statistics: 该类别的统计信息
    """
    category_name: str
    data_pairs: List[DataPair]
    scores: Optional[List[float]] = None
    statistics: Optional[Dict] = None
    
    def __len__(self):
        return len(self.data_pairs)


@dataclass
class BenchmarkData:
    """
    完整的Benchmark数据集
    
    Attributes:
        categories: 按类别组织的数据字典
        total_pairs: 总数据对数量
        category_names: 所有类别名称列表
        metadata: Benchmark元数据
    """
    categories: Dict[str, CategoryData]
    total_pairs: int
    category_names: List[str]
    metadata: Optional[Dict] = None
    
    def get_category(self, category_name: str) -> Optional[CategoryData]:
        """获取指定类别的数据"""
        return self.categories.get(category_name)
    
    def get_all_pairs(self) -> List[DataPair]:
        """获取所有数据对"""
        all_pairs = []
        for category_data in self.categories.values():
            all_pairs.extend(category_data.data_pairs)
        return all_pairs


