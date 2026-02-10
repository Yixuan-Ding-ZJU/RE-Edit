"""
Iterative Refinement Data Types
迭代优化数据类型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from PIL import Image
from .data_types import DataPair, CategoryData


@dataclass
class IterativeDataPair(DataPair):
    """
    迭代优化数据对（扩展版）
    
    继承自DataPair，添加迭代优化所需的字段：
    - 初次编辑结果和评分
    - MLLM分析结果（CoT推理 + Re-edit指令）
    - 二次编辑结果和评分
    
    向后兼容：
    - edited_image 优先返回二次编辑结果，fallback到初次编辑
    - score 优先返回二次编辑分数，fallback到初次编辑
    """
    
    # ===== 初次编辑 (Primary Editing) =====
    primary_edited_image: Optional[Image.Image] = None
    primary_score: Optional[str] = None  # "yes" or "no"
    primary_pq_score: Optional[List[float]] = None  # PQ指标: [naturalness, artifacts]
    primary_pq_reasoning: Optional[str] = None  # PQ指标reasoning
    primary_sc_score: Optional[List[float]] = None  # SC指标: [editing_success, degree_of_overediting]
    primary_sc_reasoning: Optional[str] = None  # SC指标reasoning
    primary_if_score: Optional[float] = None  # Instruction Following指标: 0-10单分
    primary_if_reasoning: Optional[str] = None  # IF指标reasoning
    
    # ===== MLLM分析 (MLLM Analysis) =====
    cot_reasoning: Optional[str] = None  # Chain of Thought推理过程
    re_edit_instruction: Optional[str] = None  # 改进编辑指令（拼接后的字符串，向后兼容）
    re_edit_instructions: Optional[List[str]] = None  # 独立的re-edit指令列表（用于迭代refinement）
    mllm_raw_output: Optional[str] = None  # MLLM原始输出（用于调试）
    
    # ===== 二次编辑 (Refinement Editing) =====
    refined_edited_image: Optional[Image.Image] = None
    refined_score: Optional[str] = None  # "yes" or "no"
    refined_pq_score: Optional[List[float]] = None  # PQ指标: [naturalness, artifacts]
    refined_pq_reasoning: Optional[str] = None  # PQ指标reasoning
    refined_sc_score: Optional[List[float]] = None  # SC指标: [editing_success, degree_of_overediting]
    refined_sc_reasoning: Optional[str] = None  # SC指标reasoning
    refined_if_score: Optional[float] = None  # Instruction Following指标: 0-10单分
    refined_if_reasoning: Optional[str] = None  # IF指标reasoning
    refinement_iterations: Optional[int] = None  # 记录refinement迭代次数
    
    # ===== 统计信息 =====
    improvement_rate: Optional[float] = None  # 改进率：从no→yes为100%，yes→yes为0%，yes→no为-100%
    
    # ===== 向后兼容属性 =====
    @property
    def edited_image(self) -> Optional[Image.Image]:
        """
        向后兼容：优先返回二次编辑结果，fallback到初次编辑
        
        这样在不支持迭代优化的代码中，仍能正常工作
        """
        return self.refined_edited_image or self.primary_edited_image
    
    @edited_image.setter
    def edited_image(self, value: Optional[Image.Image]):
        """
        向后兼容：设置编辑结果时，同时设置primary_edited_image
        """
        self.primary_edited_image = value
    
    @property
    def score(self) -> Optional[str]:
        """
        向后兼容：优先返回二次编辑分数，fallback到初次编辑
        """
        return self.refined_score or self.primary_score
    
    @score.setter
    def score(self, value: Optional[str]):
        """
        向后兼容：设置分数时，同时设置primary_score
        """
        self.primary_score = value
    
    def calculate_improvement_rate(self) -> Optional[float]:
        """
        计算改进率（适配yes/no逻辑）
        
        改进率定义：
        - no → yes: 100% (改进)
        - yes → yes: 0% (保持)
        - yes → no: -100% (退步)
        - no → no: 0% (无变化)
        
        Returns:
            改进率 (%) 如果两次分数都存在，否则None
        """
        if self.primary_score is not None and self.refined_score is not None:
            primary = self.primary_score.lower() if isinstance(self.primary_score, str) else self.primary_score
            refined = self.refined_score.lower() if isinstance(self.refined_score, str) else self.refined_score
            
            # 确保是yes/no值
            if primary not in ["yes", "no"] or refined not in ["yes", "no"]:
                return None
            
            # 计算改进率
            if primary == "no" and refined == "yes":
                self.improvement_rate = 100.0  # 改进
            elif primary == "yes" and refined == "yes":
                self.improvement_rate = 0.0  # 保持
            elif primary == "yes" and refined == "no":
                self.improvement_rate = -100.0  # 退步
            elif primary == "no" and refined == "no":
                self.improvement_rate = 0.0  # 无变化
            
                return self.improvement_rate
        return None
    
    def has_improvement(self) -> bool:
        """
        判断是否有改进（适配yes/no逻辑）
        
        改进定义：
        - no → yes: 有改进
        - yes → yes: 无改进（保持）
        - yes → no: 无改进（退步）
        - no → no: 无改进（无变化）
        
        Returns:
            True如果从no变为yes，否则False
        """
        if self.primary_score is not None and self.refined_score is not None:
            primary = self.primary_score.lower() if isinstance(self.primary_score, str) else self.primary_score
            refined = self.refined_score.lower() if isinstance(self.refined_score, str) else self.refined_score
            
            # 确保是yes/no值
            if primary not in ["yes", "no"] or refined not in ["yes", "no"]:
                return False
            
            # 只有从no→yes才算改进
            return primary == "no" and refined == "yes"
        return False
    
    def to_dict(self) -> Dict:
        """
        转换为字典格式（用于保存和调试）
        
        Returns:
            包含所有信息的字典
        """
        return {
            "pair_id": self.pair_id,
            "category": self.category,
            "edit_instruction": self.edit_instruction,
            "original_description": self.original_description,
            "primary_score": self.primary_score,
            "primary_pq_score": self.primary_pq_score,
            "primary_pq_reasoning": self.primary_pq_reasoning,
            "primary_sc_score": self.primary_sc_score,
            "primary_sc_reasoning": self.primary_sc_reasoning,
            "refined_score": self.refined_score,
            "refined_pq_score": self.refined_pq_score,
            "refined_pq_reasoning": self.refined_pq_reasoning,
            "refined_sc_score": self.refined_sc_score,
            "refined_sc_reasoning": self.refined_sc_reasoning,
            "primary_if_score": self.primary_if_score,
            "primary_if_reasoning": self.primary_if_reasoning,
            "refined_if_score": self.refined_if_score,
            "refined_if_reasoning": self.refined_if_reasoning,
            "improvement_rate": self.improvement_rate,
            "cot_reasoning": self.cot_reasoning,
            "re_edit_instruction": self.re_edit_instruction,
            "has_improvement": self.has_improvement()
        }


@dataclass
class IterativeCategoryData(CategoryData):
    """
    迭代优化类别数据（扩展版）
    
    继承自CategoryData，添加迭代优化的统计信息
    """
    
    # 扩展统计信息
    primary_scores: Optional[List[str]] = None  # 初次编辑分数列表（"yes"/"no"）
    refined_scores: Optional[List[str]] = None  # 二次编辑分数列表（"yes"/"no"）
    improvement_rates: Optional[List[float]] = None  # 改进率列表
    primary_pq_scores: Optional[List[List[float]]] = None  # 初次编辑PQ分数列表 [[naturalness, artifacts], ...]
    refined_pq_scores: Optional[List[List[float]]] = None  # 二次编辑PQ分数列表 [[naturalness, artifacts], ...]
    primary_sc_scores: Optional[List[List[float]]] = None  # 初次编辑SC分数列表 [[editing_success, degree_of_overediting], ...]
    refined_sc_scores: Optional[List[List[float]]] = None  # 二次编辑SC分数列表 [[editing_success, degree_of_overediting], ...]
    primary_if_scores: Optional[List[float]] = None  # 初次编辑IF分数列表 [0-10, ...]
    refined_if_scores: Optional[List[float]] = None  # 二次编辑IF分数列表 [0-10, ...]
    
    def calculate_statistics(self):
        """
        计算类别级别的统计信息（适配yes/no逻辑）
        """
        if not self.data_pairs:
            return
        
        # 提取分数
        self.primary_scores = []
        self.refined_scores = []
        self.improvement_rates = []
        self.primary_pq_scores = []
        self.refined_pq_scores = []
        self.primary_sc_scores = []
        self.refined_sc_scores = []
        self.primary_if_scores = []
        self.refined_if_scores = []
        
        for pair in self.data_pairs:
            if isinstance(pair, IterativeDataPair):
                if pair.primary_score is not None:
                    self.primary_scores.append(pair.primary_score)
                if pair.refined_score is not None:
                    self.refined_scores.append(pair.refined_score)
                if pair.improvement_rate is not None:
                    self.improvement_rates.append(pair.improvement_rate)
                if pair.primary_pq_score is not None:
                    self.primary_pq_scores.append(pair.primary_pq_score)
                if pair.refined_pq_score is not None:
                    self.refined_pq_scores.append(pair.refined_pq_score)
                if pair.primary_sc_score is not None:
                    self.primary_sc_scores.append(pair.primary_sc_score)
                if pair.refined_sc_score is not None:
                    self.refined_sc_scores.append(pair.refined_sc_score)
                if pair.primary_if_score is not None:
                    self.primary_if_scores.append(pair.primary_if_score)
                if pair.refined_if_score is not None:
                    self.refined_if_scores.append(pair.refined_if_score)
        
        # 计算yes/no统计（确保处理str类型）
        primary_yes_count = sum(1 for s in self.primary_scores if s and isinstance(s, str) and s.lower() == "yes")
        primary_no_count = len(self.primary_scores) - primary_yes_count
        primary_yes_rate = (primary_yes_count / len(self.primary_scores) * 100) if self.primary_scores else 0.0
        
        refined_yes_count = sum(1 for s in self.refined_scores if s and isinstance(s, str) and s.lower() == "yes")
        refined_no_count = len(self.refined_scores) - refined_yes_count
        refined_yes_rate = (refined_yes_count / len(self.refined_scores) * 100) if self.refined_scores else 0.0
        
        # 计算改进/保持/退步/无变化的统计
        improved_count = 0      # no → yes
        maintained_count = 0    # yes → yes
        regression_count = 0    # yes → no
        unchanged_count = 0     # no → no
        
        for pair in self.data_pairs:
            if isinstance(pair, IterativeDataPair):
                if pair.primary_score is not None and pair.refined_score is not None:
                    primary = pair.primary_score.lower() if isinstance(pair.primary_score, str) else pair.primary_score
                    refined = pair.refined_score.lower() if isinstance(pair.refined_score, str) else pair.refined_score
                    
                    if primary in ["yes", "no"] and refined in ["yes", "no"]:
                        if primary == "no" and refined == "yes":
                            improved_count += 1
                        elif primary == "yes" and refined == "yes":
                            maintained_count += 1
                        elif primary == "yes" and refined == "no":
                            regression_count += 1
                        elif primary == "no" and refined == "no":
                            unchanged_count += 1
        
        total_valid_pairs = improved_count + maintained_count + regression_count + unchanged_count
        
        # 计算各种比率
        improvement_rate = (improved_count / total_valid_pairs * 100) if total_valid_pairs > 0 else 0.0
        maintained_rate = (maintained_count / total_valid_pairs * 100) if total_valid_pairs > 0 else 0.0
        regression_rate = (regression_count / total_valid_pairs * 100) if total_valid_pairs > 0 else 0.0
        unchanged_rate = (unchanged_count / total_valid_pairs * 100) if total_valid_pairs > 0 else 0.0
        
        # 计算统计信息（保持原有字段结构以兼容报告生成）
        # 计算 PQ 和 SC 的最小值统计
        primary_pq_min_scores = [min(score) for score in self.primary_pq_scores] if self.primary_pq_scores else []
        refined_pq_min_scores = [min(score) for score in self.refined_pq_scores] if self.refined_pq_scores else []
        primary_sc_min_scores = [min(score) for score in self.primary_sc_scores] if self.primary_sc_scores else []
        refined_sc_min_scores = [min(score) for score in self.refined_sc_scores] if self.refined_sc_scores else []
        
        primary_pq_avg = sum(primary_pq_min_scores) / len(primary_pq_min_scores) if primary_pq_min_scores else 0.0
        refined_pq_avg = sum(refined_pq_min_scores) / len(refined_pq_min_scores) if refined_pq_min_scores else 0.0
        primary_sc_avg = sum(primary_sc_min_scores) / len(primary_sc_min_scores) if primary_sc_min_scores else 0.0
        refined_sc_avg = sum(refined_sc_min_scores) / len(refined_sc_min_scores) if refined_sc_min_scores else 0.0
        
        self.statistics = {
            # Yes/No统计（新增）
            "primary_yes_count": primary_yes_count,
            "primary_no_count": primary_no_count,
            "primary_yes_rate": primary_yes_rate,
            "refined_yes_count": refined_yes_count,
            "refined_no_count": refined_no_count,
            "refined_yes_rate": refined_yes_rate,
            # 改进/保持/退步/无变化统计（新增）
            "improved_count": improved_count,
            "improvement_rate": improvement_rate,
            "maintained_count": maintained_count,
            "maintained_rate": maintained_rate,
            "regression_count": regression_count,
            "regression_rate": regression_rate,
            "unchanged_count": unchanged_count,
            "unchanged_rate": unchanged_rate,
            # 原有字段（为了兼容，使用yes_rate替代avg）
            "primary_avg": primary_yes_rate,  # 使用yes_rate作为"平均分"
            "refined_avg": refined_yes_rate,  # 使用yes_rate作为"平均分"
            "avg_improvement_rate": sum(self.improvement_rates) / len(self.improvement_rates) if self.improvement_rates else 0,
            "total_count": len(self.data_pairs),
            # PQ指标统计（使用最小值）
            "primary_pq_avg": primary_pq_avg,
            "refined_pq_avg": refined_pq_avg,
            "pq_improvement": refined_pq_avg - primary_pq_avg,
            "pq_improvement_rate": ((refined_pq_avg - primary_pq_avg) / primary_pq_avg * 100) if primary_pq_avg > 0 else 0.0,
            # PQ指标详细统计（保留原有的naturalness和artifacts平均值）
            "primary_pq_avg_naturalness": sum(s[0] for s in self.primary_pq_scores) / len(self.primary_pq_scores) if self.primary_pq_scores else 0.0,
            "primary_pq_avg_artifacts": sum(s[1] for s in self.primary_pq_scores) / len(self.primary_pq_scores) if self.primary_pq_scores else 0.0,
            "refined_pq_avg_naturalness": sum(s[0] for s in self.refined_pq_scores) / len(self.refined_pq_scores) if self.refined_pq_scores else 0.0,
            "refined_pq_avg_artifacts": sum(s[1] for s in self.refined_pq_scores) / len(self.refined_pq_scores) if self.refined_pq_scores else 0.0,
            "pq_improvement_naturalness": (sum(s[0] for s in self.refined_pq_scores) / len(self.refined_pq_scores) - 
                                         sum(s[0] for s in self.primary_pq_scores) / len(self.primary_pq_scores)) if (self.primary_pq_scores and self.refined_pq_scores) else 0.0,
            "pq_improvement_artifacts": (sum(s[1] for s in self.refined_pq_scores) / len(self.refined_pq_scores) - 
                                        sum(s[1] for s in self.primary_pq_scores) / len(self.primary_pq_scores)) if (self.primary_pq_scores and self.refined_pq_scores) else 0.0,
            # SC指标统计（使用最小值）
            "primary_sc_avg": primary_sc_avg,
            "refined_sc_avg": refined_sc_avg,
            "sc_improvement": refined_sc_avg - primary_sc_avg,
            "sc_improvement_rate": ((refined_sc_avg - primary_sc_avg) / primary_sc_avg * 100) if primary_sc_avg > 0 else 0.0,
            # SC指标详细统计（保留原有的editing_success和overediting平均值）
            "primary_sc_avg_editing_success": sum(s[0] for s in self.primary_sc_scores) / len(self.primary_sc_scores) if self.primary_sc_scores else 0.0,
            "primary_sc_avg_overediting": sum(s[1] for s in self.primary_sc_scores) / len(self.primary_sc_scores) if self.primary_sc_scores else 0.0,
            "refined_sc_avg_editing_success": sum(s[0] for s in self.refined_sc_scores) / len(self.refined_sc_scores) if self.refined_sc_scores else 0.0,
            "refined_sc_avg_overediting": sum(s[1] for s in self.refined_sc_scores) / len(self.refined_sc_scores) if self.refined_sc_scores else 0.0,
            "sc_improvement_editing_success": (sum(s[0] for s in self.refined_sc_scores) / len(self.refined_sc_scores) - 
                                             sum(s[0] for s in self.primary_sc_scores) / len(self.primary_sc_scores)) if (self.primary_sc_scores and self.refined_sc_scores) else 0.0,
            "sc_improvement_overediting": (sum(s[1] for s in self.refined_sc_scores) / len(self.refined_sc_scores) - 
                                         sum(s[1] for s in self.primary_sc_scores) / len(self.primary_sc_scores)) if (self.primary_sc_scores and self.refined_sc_scores) else 0.0,
            # IF指标统计（Instruction Following: 0-10单分）
            "primary_if_avg": sum(self.primary_if_scores) / len(self.primary_if_scores) if self.primary_if_scores else 0.0,
            "refined_if_avg": sum(self.refined_if_scores) / len(self.refined_if_scores) if self.refined_if_scores else 0.0,
            "if_improvement": (sum(self.refined_if_scores) / len(self.refined_if_scores) - 
                              sum(self.primary_if_scores) / len(self.primary_if_scores)) if (self.primary_if_scores and self.refined_if_scores) else 0.0,
            "if_improvement_rate": ((sum(self.refined_if_scores) / len(self.refined_if_scores) - 
                                    sum(self.primary_if_scores) / len(self.primary_if_scores)) / 
                                   (sum(self.primary_if_scores) / len(self.primary_if_scores)) * 100) if (self.primary_if_scores and self.refined_if_scores and sum(self.primary_if_scores) > 0) else 0.0
        }
        
        # 向后兼容
        self.scores = self.refined_scores if self.refined_scores else self.primary_scores


@dataclass
class IterativeBenchmarkData:
    """
    迭代优化Benchmark数据集
    
    包含所有类别的迭代优化数据
    """
    categories: Dict[str, IterativeCategoryData]
    total_pairs: int
    category_names: List[str]
    metadata: Optional[Dict] = None
    
    def get_category(self, category_name: str) -> Optional[IterativeCategoryData]:
        """获取指定类别的数据"""
        return self.categories.get(category_name)
    
    def get_all_pairs(self) -> List[IterativeDataPair]:
        """获取所有数据对"""
        all_pairs = []
        for category_data in self.categories.values():
            all_pairs.extend(category_data.data_pairs)
        return all_pairs
    
    def calculate_global_statistics(self) -> Dict:
        """
        计算全局统计信息（适配yes/no逻辑，并添加PQ和SC指标统计）
        
        Returns:
            全局统计字典
        """
        all_primary_scores = []
        all_refined_scores = []
        all_improvement_rates = []
        total_improved = 0
        
        # 收集 PQ、SC 和 IF 指标数据
        all_primary_pq_scores = []
        all_refined_pq_scores = []
        all_primary_sc_scores = []
        all_refined_sc_scores = []
        all_primary_if_scores = []
        all_refined_if_scores = []
        
        for category_data in self.categories.values():
            if category_data.primary_scores:
                all_primary_scores.extend(category_data.primary_scores)
            if category_data.refined_scores:
                all_refined_scores.extend(category_data.refined_scores)
            if category_data.improvement_rates:
                all_improvement_rates.extend(category_data.improvement_rates)
            if category_data.statistics:
                total_improved += category_data.statistics.get("improved_count", 0)
            
            # 收集 PQ 和 SC 分数
            if category_data.primary_pq_scores:
                all_primary_pq_scores.extend(category_data.primary_pq_scores)
            if category_data.refined_pq_scores:
                all_refined_pq_scores.extend(category_data.refined_pq_scores)
            if category_data.primary_sc_scores:
                all_primary_sc_scores.extend(category_data.primary_sc_scores)
            if category_data.refined_sc_scores:
                all_refined_sc_scores.extend(category_data.refined_sc_scores)
            if category_data.primary_if_scores:
                all_primary_if_scores.extend(category_data.primary_if_scores)
            if category_data.refined_if_scores:
                all_refined_if_scores.extend(category_data.refined_if_scores)
        
        # 计算全局yes/no统计（确保处理str类型）
        global_primary_yes_count = sum(1 for s in all_primary_scores if s and isinstance(s, str) and s.lower() == "yes")
        global_primary_yes_rate = (global_primary_yes_count / len(all_primary_scores) * 100) if all_primary_scores else 0.0
        
        global_refined_yes_count = sum(1 for s in all_refined_scores if s and isinstance(s, str) and s.lower() == "yes")
        global_refined_yes_rate = (global_refined_yes_count / len(all_refined_scores) * 100) if all_refined_scores else 0.0
        
        # 计算全局改进/保持/退步/无变化统计
        global_improved_count = 0      # no → yes
        global_maintained_count = 0    # yes → yes
        global_regression_count = 0    # yes → no
        global_unchanged_count = 0     # no → no
        
        for category_data in self.categories.values():
            if category_data.statistics:
                global_improved_count += category_data.statistics.get("improved_count", 0)
                global_maintained_count += category_data.statistics.get("maintained_count", 0)
                global_regression_count += category_data.statistics.get("regression_count", 0)
                global_unchanged_count += category_data.statistics.get("unchanged_count", 0)
        
        global_total_valid_pairs = global_improved_count + global_maintained_count + global_regression_count + global_unchanged_count
        
        # 计算全局各种比率
        global_improvement_rate = (global_improved_count / global_total_valid_pairs * 100) if global_total_valid_pairs > 0 else 0.0
        global_maintained_rate = (global_maintained_count / global_total_valid_pairs * 100) if global_total_valid_pairs > 0 else 0.0
        global_regression_rate = (global_regression_count / global_total_valid_pairs * 100) if global_total_valid_pairs > 0 else 0.0
        global_unchanged_rate = (global_unchanged_count / global_total_valid_pairs * 100) if global_total_valid_pairs > 0 else 0.0
        
        # 计算 PQ 指标全局统计（使用列表中最小值）
        primary_pq_min_scores = [min(score) for score in all_primary_pq_scores] if all_primary_pq_scores else []
        refined_pq_min_scores = [min(score) for score in all_refined_pq_scores] if all_refined_pq_scores else []
        
        global_primary_pq_avg = sum(primary_pq_min_scores) / len(primary_pq_min_scores) if primary_pq_min_scores else 0.0
        global_refined_pq_avg = sum(refined_pq_min_scores) / len(refined_pq_min_scores) if refined_pq_min_scores else 0.0
        global_pq_improvement = global_refined_pq_avg - global_primary_pq_avg
        
        # 计算 SC 指标全局统计（使用列表中最小值）
        primary_sc_min_scores = [min(score) for score in all_primary_sc_scores] if all_primary_sc_scores else []
        refined_sc_min_scores = [min(score) for score in all_refined_sc_scores] if all_refined_sc_scores else []
        
        global_primary_sc_avg = sum(primary_sc_min_scores) / len(primary_sc_min_scores) if primary_sc_min_scores else 0.0
        global_refined_sc_avg = sum(refined_sc_min_scores) / len(refined_sc_min_scores) if refined_sc_min_scores else 0.0
        global_sc_improvement = global_refined_sc_avg - global_primary_sc_avg
        
        # 计算 IF 指标全局统计（Instruction Following: 0-10单分）
        global_primary_if_avg = sum(all_primary_if_scores) / len(all_primary_if_scores) if all_primary_if_scores else 0.0
        global_refined_if_avg = sum(all_refined_if_scores) / len(all_refined_if_scores) if all_refined_if_scores else 0.0
        global_if_improvement = global_refined_if_avg - global_primary_if_avg
        
        return {
            # Yes/No统计（新增）
            "global_primary_yes_count": global_primary_yes_count,
            "global_primary_yes_rate": global_primary_yes_rate,
            "global_refined_yes_count": global_refined_yes_count,
            "global_refined_yes_rate": global_refined_yes_rate,
            # 改进/保持/退步/无变化统计（新增）
            "global_improved_count": global_improved_count,
            "global_improvement_rate": global_improvement_rate,
            "global_maintained_count": global_maintained_count,
            "global_maintained_rate": global_maintained_rate,
            "global_regression_count": global_regression_count,
            "global_regression_rate": global_regression_rate,
            "global_unchanged_count": global_unchanged_count,
            "global_unchanged_rate": global_unchanged_rate,
            # 原有字段（为了兼容，使用yes_rate替代avg）
            "global_primary_avg": global_primary_yes_rate,  # 使用yes_rate作为"平均分"
            "global_refined_avg": global_refined_yes_rate,  # 使用yes_rate作为"平均分"
            "global_avg_improvement_rate": sum(all_improvement_rates) / len(all_improvement_rates) if all_improvement_rates else 0,
            "total_improved_count": global_improved_count,  # 使用新的统计值
            "total_pairs": self.total_pairs,
            "improvement_percentage": global_improvement_rate,  # 使用新的统计值
            # PQ 指标全局统计（新增）
            "global_primary_pq_avg": global_primary_pq_avg,
            "global_refined_pq_avg": global_refined_pq_avg,
            "global_pq_improvement": global_pq_improvement,
            "global_pq_improvement_rate": (global_pq_improvement / global_primary_pq_avg * 100) if global_primary_pq_avg > 0 else 0.0,
            # SC 指标全局统计（新增）
            "global_primary_sc_avg": global_primary_sc_avg,
            "global_refined_sc_avg": global_refined_sc_avg,
            "global_sc_improvement": global_sc_improvement,
            "global_sc_improvement_rate": (global_sc_improvement / global_primary_sc_avg * 100) if global_primary_sc_avg > 0 else 0.0,
            # IF 指标全局统计（Instruction Following）
            "global_primary_if_avg": global_primary_if_avg,
            "global_refined_if_avg": global_refined_if_avg,
            "global_if_improvement": global_if_improvement,
            "global_if_improvement_rate": (global_if_improvement / global_primary_if_avg * 100) if global_primary_if_avg > 0 else 0.0
        }




