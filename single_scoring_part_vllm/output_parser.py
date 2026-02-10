# -*- coding: utf-8 -*-
"""
模型输出解析模块
解析JSON格式的输出，支持score为数字列表或yes/no字符串
"""

import json
import re
from typing import Dict, List, Optional, Union, Any
import logging


class OutputParser:
    """输出解析器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def parse_output(self, generated_text: str) -> Dict[str, Any]:
        """
        解析模型生成的文本，提取JSON格式的输出
        
        Args:
            generated_text: 模型生成的文本
            
        Returns:
            解析后的字典，包含score和reasoning字段
            如果解析失败，返回 {"score": None, "reasoning": "", "error": "..."}
        """
        try:
            # 尝试提取JSON部分
            json_str = self._extract_json(generated_text)
            
            if json_str:
                parsed = json.loads(json_str)
                
                # 验证必要字段
                if "score" not in parsed:
                    self.logger.warning("No 'score' field in parsed JSON")
                    return {
                        "score": None,
                        "reasoning": parsed.get("reasoning", parsed.get("reason", "")),
                        "error": "Missing 'score' field"
                    }
                
                # 支持"reasoning"和"reason"两种字段名（instruction_following使用"reason"）
                reasoning = parsed.get("reasoning", parsed.get("reason", ""))
                
                return {
                    "score": parsed["score"],
                    "reasoning": reasoning,
                    "raw_output": generated_text
                }
            else:
                self.logger.warning("No JSON found in generated text")
                return {
                    "score": None,
                    "reasoning": "",
                    "error": "No JSON found in output",
                    "raw_output": generated_text
                }
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return {
                "score": None,
                "reasoning": "",
                "error": f"JSON decode error: {str(e)}",
                "raw_output": generated_text
            }
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return {
                "score": None,
                "reasoning": "",
                "error": f"Parse error: {str(e)}",
                "raw_output": generated_text
            }
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        从文本中提取JSON部分
        
        Args:
            text: 原始文本
            
        Returns:
            提取的JSON字符串，如果未找到返回None
        """
        # 方法1: 尝试找到 {...} 格式的JSON
        # 使用正则表达式匹配最外层的大括号
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            # 选择最长的匹配（可能是最完整的JSON）
            json_str = max(matches, key=len)
            
            # 验证是否是有效的JSON
            try:
                json.loads(json_str)
                return json_str
            except:
                pass
        
        # 方法2: 尝试找到代码块中的JSON
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            try:
                json.loads(matches[0])
                return matches[0]
            except:
                pass
        
        # 方法3: 尝试直接解析整个文本
        text_stripped = text.strip()
        if text_stripped.startswith('{') and text_stripped.endswith('}'):
            try:
                json.loads(text_stripped)
                return text_stripped
            except:
                pass
        
        return None
    
    def normalize_score(self, score: Any) -> Union[str, List[float], None]:
        """
        标准化score值
        
        Args:
            score: 原始score值，可能是 "yes"/"no" 或数字列表
            
        Returns:
            标准化后的score值
        """
        if score is None:
            return None
        
        # 如果是字符串，检查是否是yes/no
        if isinstance(score, str):
            score_lower = score.lower().strip()
            if score_lower in ['yes', 'no']:
                return score_lower
            # 尝试转换为数字
            try:
                return float(score)
            except:
                return score
        
        # 如果是列表，转换为浮点数列表
        if isinstance(score, list):
            try:
                return [float(x) for x in score]
            except:
                return score
        
        # 如果是数字，直接返回
        if isinstance(score, (int, float)):
            return float(score)
        
        return score
    
    def parse_batch_outputs(self, outputs: List) -> List[Dict[str, Any]]:
        """
        批量解析输出
        
        Args:
            outputs: vLLM生成的结果列表
            
        Returns:
            解析后的结果列表
        """
        parsed_results = []
        
        for output in outputs:
            generated_text = output.outputs[0].text
            parsed = self.parse_output(generated_text)
            
            # 标准化score
            if parsed["score"] is not None:
                parsed["score"] = self.normalize_score(parsed["score"])
            
            parsed_results.append(parsed)
        
        return parsed_results

