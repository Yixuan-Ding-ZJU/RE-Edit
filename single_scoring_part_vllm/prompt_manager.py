# -*- coding: utf-8 -*-
"""
Prompt配置和管理模块
支持从YAML配置文件读取prompt模板
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, List
import logging
import re


class PromptManager:
    """Prompt管理器"""
    
    def __init__(self, config_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        初始化Prompt管理器
        
        Args:
            config_path: YAML配置文件路径
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.prompts = {}
        
        if config_path:
            self.load_from_config(config_path)
    
    def load_from_config(self, config_path: str):
        """
        从YAML配置文件加载prompt配置
        
        Args:
            config_path: 配置文件路径
        """
        self.logger.info(f"Loading prompts from config: {config_path}")
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 从config中提取prompts配置
        if 'prompts' in config:
            self.prompts = config['prompts']
            self.logger.info(f"Loaded prompts for categories: {list(self.prompts.keys())}")
        else:
            self.logger.warning("No 'prompts' section found in config")
    
    def get_category_prompt(self, category: str) -> Optional[Dict[str, str]]:
        """
        获取指定类别的prompt配置
        
        Args:
            category: 类别名称（物理、环境、社会、因果、指代）
            
        Returns:
            包含system_prompt和user_prompt_template的字典，如果不存在返回None
        """
        if category in self.prompts:
            return self.prompts[category]
        return None
    
    def get_metric_prompt(self, metric_name: str) -> Optional[Dict[str, str]]:
        """
        获取指定metric的prompt配置
        
        Args:
            metric_name: metric名称（如 "pq_metric", "sc_metric"）
            
        Returns:
            包含system_prompt和user_prompt_template的字典，如果不存在返回None
        """
        if metric_name in self.prompts:
            return self.prompts[metric_name]
        return None
    
    def format_user_prompt(self,
                          template: str,
                          edit_instruction: str = "",
                          original_description: str = "",
                          rationale: str = "",
                          **kwargs) -> str:
        """
        格式化user prompt模板
        
        Args:
            template: 模板字符串，支持 {edit_instruction}, {original_description}, {rationale} 等占位符
            edit_instruction: 编辑指令
            original_description: 原始描述
            rationale: 任务难点说明
            **kwargs: 其他模板变量
            
        Returns:
            格式化后的prompt字符串
        """
        # 准备所有模板变量
        template_vars = {
            'edit_instruction': edit_instruction,
            'original_description': original_description,
            'rationale': rationale,
            **kwargs
        }
        
        # 替换模板变量
        formatted = template
        
        # 使用双大括号转义，然后替换单大括号
        # YAML中的模板使用 {{ }} 表示字面量 { }
        # 我们需要先处理 {{ }}，然后替换 { }
        
        # 先替换 {{ 和 }} 为临时标记
        formatted = formatted.replace('{{', '__DOUBLE_OPEN__')
        formatted = formatted.replace('}}', '__DOUBLE_CLOSE__')
        
        # 替换单大括号变量
        for key, value in template_vars.items():
            placeholder = '{' + key + '}'
            formatted = formatted.replace(placeholder, str(value))
        
        # 恢复双大括号（用于JSON格式输出）
        formatted = formatted.replace('__DOUBLE_OPEN__', '{')
        formatted = formatted.replace('__DOUBLE_CLOSE__', '}')
        
        return formatted
    
    def build_messages(self,
                      system_prompt: str,
                      user_prompt: str,
                      images: List[str]) -> List[Dict]:
        """
        构建消息格式
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词（格式化后的）
            images: 图片列表（URL或base64）
            
        Returns:
            消息列表，格式符合vLLM要求
        """
        # 构建用户消息内容
        user_content = []
        
        # 添加图片
        for img in images:
            user_content.append({
                "type": "image",
                "image": img
            })
        
        # 合并system prompt和user prompt
        # 根据Qwen3-VL的格式，将system prompt添加到文本开头
        if system_prompt:
            full_user_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_user_prompt = user_prompt
        
        # 添加文本
        user_content.append({
            "type": "text",
            "text": full_user_prompt
        })
        
        # 构建消息（Qwen3-VL使用user role）
        messages = [{
            "role": "user",
            "content": user_content
        }]
        
        return messages

