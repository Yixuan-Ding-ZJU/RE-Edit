# -*- coding: utf-8 -*-
"""
评分结果统计模块
生成统计总结JSON文件
"""

from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict
from datetime import datetime


class StatisticsCalculator:
    """统计计算器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_category_statistics(self,
                                     results: List[Dict],
                                     category: str) -> Dict[str, Any]:
        """
        计算单个类别的统计信息
        
        Args:
            results: 评分结果列表，每个元素包含category、score等信息
            category: 类别名称
            
        Returns:
            统计信息字典
        """
        category_results = [r for r in results if r.get("category") == category]
        
        if not category_results:
            return {}
        
        stats = {
            "total_count": len(category_results),
            "category": category
        }
        
        # 分析score类型
        first_score = category_results[0].get("score")
        
        if isinstance(first_score, str):
            # yes/no类型
            yes_count = sum(1 for r in category_results if r.get("score") == "yes")
            no_count = sum(1 for r in category_results if r.get("score") == "no")
            yes_rate = (yes_count / len(category_results) * 100) if category_results else 0
            
            stats.update({
                "yes_count": yes_count,
                "no_count": no_count,
                "yes_rate": yes_rate,
                "average": yes_rate,
                "std": 0.0,
                "min": 0.0,
                "max": 100.0
            })
        
        elif isinstance(first_score, list):
            # 数字列表类型（如PQ、SC指标）
            if len(first_score) == 2:
                # 两个分数的列表
                scores_0 = [r.get("score", [0, 0])[0] for r in category_results if isinstance(r.get("score"), list) and len(r.get("score", [])) > 0]
                scores_1 = [r.get("score", [0, 0])[1] for r in category_results if isinstance(r.get("score"), list) and len(r.get("score", [])) > 1]
                
                if scores_0:
                    stats["avg_0"] = sum(scores_0) / len(scores_0)
                    stats["min_0"] = min(scores_0)
                    stats["max_0"] = max(scores_0)
                
                if scores_1:
                    stats["avg_1"] = sum(scores_1) / len(scores_1)
                    stats["min_1"] = min(scores_1)
                    stats["max_1"] = max(scores_1)
                
                # 计算平均分（两个分数的平均值）
                avg_scores = [(s[0] + s[1]) / 2 for s in [r.get("score", [0, 0]) for r in category_results] if isinstance(s, list) and len(s) == 2]
                if avg_scores:
                    stats["average"] = sum(avg_scores) / len(avg_scores)
                    stats["std"] = self._calculate_std(avg_scores)
                    stats["min"] = min(avg_scores)
                    stats["max"] = max(avg_scores)
        
        elif isinstance(first_score, (int, float)):
            # 单个数字（如instruction_following的0-10分数）
            scores = [r.get("score") for r in category_results if isinstance(r.get("score"), (int, float))]
            if scores:
                stats["average"] = sum(scores) / len(scores)
                stats["std"] = self._calculate_std(scores)
                stats["min"] = min(scores)
                stats["max"] = max(scores)
        
        return stats
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def calculate_overall_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        计算整体统计信息
        
        Args:
            results: 所有评分结果
            
        Returns:
            整体统计信息字典
        """
        if not results:
            return {}
        
        stats = {
            "total_samples": len(results)
        }
        
        # 分析score类型
        first_score = results[0].get("score")
        
        if isinstance(first_score, str):
            # yes/no类型
            yes_count = sum(1 for r in results if r.get("score") == "yes")
            no_count = sum(1 for r in results if r.get("score") == "no")
            yes_rate = (yes_count / len(results) * 100) if results else 0
            
            stats.update({
                "yes_count": yes_count,
                "no_count": no_count,
                "yes_rate": yes_rate,
                "average": yes_rate
            })
        
        elif isinstance(first_score, list):
            # 数字列表类型
            if len(first_score) == 2:
                scores_0 = [r.get("score", [0, 0])[0] for r in results if isinstance(r.get("score"), list) and len(r.get("score", [])) > 0]
                scores_1 = [r.get("score", [0, 0])[1] for r in results if isinstance(r.get("score"), list) and len(r.get("score", [])) > 1]
                
                if scores_0:
                    stats["avg_0"] = sum(scores_0) / len(scores_0)
                if scores_1:
                    stats["avg_1"] = sum(scores_1) / len(scores_1)
                
                avg_scores = [(s[0] + s[1]) / 2 for s in [r.get("score", [0, 0]) for r in results] if isinstance(s, list) and len(s) == 2]
                if avg_scores:
                    stats["average"] = sum(avg_scores) / len(avg_scores)
        
        elif isinstance(first_score, (int, float)):
            # 单个数字（如instruction_following的0-10分数）
            scores = [r.get("score") for r in results if isinstance(r.get("score"), (int, float))]
            if scores:
                stats["average"] = sum(scores) / len(scores)
                stats["std"] = self._calculate_std(scores)
                stats["min"] = min(scores)
                stats["max"] = max(scores)
        
        return stats
    
    def generate_summary_json(self,
                             results: List[Dict],
                             categories: List[str],
                             metric_name: str,
                             metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成完整的统计总结JSON
        
        Args:
            results: 所有评分结果
            categories: 类别列表（只统计实际有数据的类别）
            metric_name: 指标名称
            metadata: 元数据（数据源、模型信息等）
            
        Returns:
            完整的统计JSON字典
        """
        # 从结果中提取实际存在的类别（避免统计不存在的类别）
        actual_categories = list(set(r.get("category") for r in results if r.get("category")))
        # 只保留在categories列表中的类别
        actual_categories = [cat for cat in actual_categories if cat in categories]
        
        if not actual_categories:
            # 如果没有实际类别，使用传入的categories（可能是全局类别）
            actual_categories = categories
        
        summary = {
            "metadata": metadata,
            "category_statistics": {},
            "overall_statistics": {},
            "summary": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 计算各类别统计（只统计实际有数据的类别）
        category_stats = {}
        for category in actual_categories:
            cat_stats = self.calculate_category_statistics(results, category)
            if cat_stats and cat_stats.get("total_count", 0) > 0:
                category_stats[category] = cat_stats
        
        summary["category_statistics"] = category_stats
        
        # 计算整体统计
        overall_stats = self.calculate_overall_statistics(results)
        summary["overall_statistics"] = overall_stats
        
        # 生成摘要（只包含实际评价的类别）
        summary["summary"] = {
            "num_categories": len(category_stats),
            "total_samples": len(results),
            "metric_name": metric_name,
            "category_means": {cat: category_stats.get(cat, {}).get("average", 0) for cat in category_stats.keys()},
            "overall_mean": overall_stats.get("average", 0)
        }
        
        # 根据metric类型添加特定的统计字段
        if results:
            first_score = results[0].get("score")
            
            if isinstance(first_score, str):
                # yes/no类型（五大维度）
                summary["summary"]["primary_yes_rate"] = overall_stats.get("yes_rate", 0)
                summary["summary"]["primary_yes_count"] = overall_stats.get("yes_count", 0)
                summary["summary"]["primary_no_count"] = overall_stats.get("no_count", 0)
                
                # 为每个类别添加yes/no统计
                for cat in category_stats:
                    if "yes_count" in category_stats[cat]:
                        category_stats[cat]["primary_yes_count"] = category_stats[cat]["yes_count"]
                        category_stats[cat]["primary_no_count"] = category_stats[cat]["no_count"]
                        category_stats[cat]["primary_yes_rate"] = category_stats[cat]["yes_rate"]
            
            elif isinstance(first_score, list) and len(first_score) == 2:
                # 数字列表类型（PQ/SC指标）
                if metric_name == "pq_metric":
                    summary["summary"]["primary_pq_avg"] = overall_stats.get("average", 0)
                    summary["summary"]["primary_pq_avg_naturalness"] = overall_stats.get("avg_0", 0)
                    summary["summary"]["primary_pq_avg_artifacts"] = overall_stats.get("avg_1", 0)
                    
                    # 为每个类别添加PQ统计
                    for cat in category_stats:
                        if "avg_0" in category_stats[cat]:
                            category_stats[cat]["primary_pq_avg"] = category_stats[cat]["average"]
                            category_stats[cat]["primary_pq_avg_naturalness"] = category_stats[cat]["avg_0"]
                            category_stats[cat]["primary_pq_avg_artifacts"] = category_stats[cat]["avg_1"]
                
                elif metric_name == "sc_metric":
                    summary["summary"]["primary_sc_avg"] = overall_stats.get("average", 0)
                    summary["summary"]["primary_sc_avg_editing_success"] = overall_stats.get("avg_0", 0)
                    summary["summary"]["primary_sc_avg_overediting"] = overall_stats.get("avg_1", 0)
                    
                    # 为每个类别添加SC统计
                    for cat in category_stats:
                        if "avg_0" in category_stats[cat]:
                            category_stats[cat]["primary_sc_avg"] = category_stats[cat]["average"]
                            category_stats[cat]["primary_sc_avg_editing_success"] = category_stats[cat]["avg_0"]
                            category_stats[cat]["primary_sc_avg_overediting"] = category_stats[cat]["avg_1"]
            
            elif isinstance(first_score, (int, float)):
                # 单个数字类型（instruction_following指标）
                if metric_name == "instruction_following":
                    summary["summary"]["primary_if_avg"] = overall_stats.get("average", 0)
                    summary["summary"]["primary_if_min"] = overall_stats.get("min", 0)
                    summary["summary"]["primary_if_max"] = overall_stats.get("max", 0)
                    summary["summary"]["primary_if_std"] = overall_stats.get("std", 0)
                    
                    # 为每个类别添加instruction_following统计
                    for cat in category_stats:
                        if "average" in category_stats[cat]:
                            category_stats[cat]["primary_if_avg"] = category_stats[cat]["average"]
                            category_stats[cat]["primary_if_min"] = category_stats[cat].get("min", 0)
                            category_stats[cat]["primary_if_max"] = category_stats[cat].get("max", 0)
                            category_stats[cat]["primary_if_std"] = category_stats[cat].get("std", 0)
        
        return summary

