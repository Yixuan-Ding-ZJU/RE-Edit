#!/usr/bin/env python3
"""
使用OpenAI GPT API作为Reward Model进行图像编辑评分

使用方法：
    python /data2/yixuan/image_edit_benchmark/src/models/reward/gpt_reward_scorer.py \
        --input-dir "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_14_flux_1_kontext" \
        --benchmark-json "/data2/yixuan/Benchmark/version_4_200_pair_human_review/version_4_simplified_v2_translated_with_images.json" \
        --config-yaml "/data2/yixuan/image_edit_benchmark/config_iterative_refinement.yaml" \
        --output "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_14_flux_1_kontext/gpt_reward_results.json" \
        --api-key "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" \
        --model "gpt-4.1-2025-04-14" \
        --log-file "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_14_flux_1_kontext/gpt_reward_results.log"  # 可选，不指定会自动生成

    python /data2/yixuan/image_edit_benchmark/src/models/reward/gpt_reward_scorer.py \
        --input-dir "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_16_qwen_image_edit" \
        --benchmark-json "/data2/yixuan/Benchmark/version_4_200_pair_human_review/version_4_simplified_v2_translated_with_images.json" \
        --config-yaml "/data2/yixuan/image_edit_benchmark/config_iterative_refinement.yaml" \
        --output "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_16_qwen_image_edit/gpt_reward_results.json" \
        --api-key "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" \
        --model "gpt-4.1-2025-04-14" \
        --log-file "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_16_qwen_image_edit/gpt_reward_results.log"  # 可选，不指定会自动生成

python /data2/yixuan/image_edit_benchmark/src/models/reward/gpt_reward_scorer.py \
        --input-dir "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_18_step1x_edit_v1p2_preview" \
        --benchmark-json "/data2/yixuan/Benchmark/version_4_200_pair_human_review/version_4_simplified_v2_translated_with_images.json" \
        --config-yaml "/data2/yixuan/image_edit_benchmark/config_iterative_refinement.yaml" \
        --output "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_18_step1x_edit_v1p2_preview/gpt_reward_results.json" \
        --api-key "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" \
        --model "gpt-4.1-2025-04-14" \
        --log-file "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_18_step1x_edit_v1p2_preview/gpt_reward_results.log"  # 可选，不指定会自动生成

python /data2/yixuan/image_edit_benchmark/src/models/reward/gpt_reward_scorer.py \
        --input-dir "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_21_step1x_edit_v1p1_diffusers" \
        --benchmark-json "/data2/yixuan/Benchmark/version_4_200_pair_human_review/version_4_simplified_v2_translated_with_images.json" \
        --config-yaml "/data2/yixuan/image_edit_benchmark/config_iterative_refinement.yaml" \
        --output "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_21_step1x_edit_v1p1_diffusers/gpt_reward_results.json" \
        --api-key "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" \
        --model "gpt-4.1-2025-04-14" \
        --log-file "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_21_step1x_edit_v1p1_diffusers/gpt_reward_results.log"  # 可选，不指定会自动生成

python /data2/yixuan/image_edit_benchmark/src/models/reward/gpt_reward_scorer.py \
        --input-dir "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_22_DreamOmni2" \
        --benchmark-json "/data2/yixuan/Benchmark/version_4_200_pair_human_review/version_4_simplified_v2_translated_with_images.json" \
        --config-yaml "/data2/yixuan/image_edit_benchmark/config_iterative_refinement.yaml" \
        --output "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_22_DreamOmni2/gpt_reward_results.json" \
        --api-key "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" \
        --model "gpt-4.1-2025-04-14" \
        --log-file "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_22_DreamOmni2/gpt_reward_results.log"  # 可选，不指定会自动生成

python /data2/yixuan/image_edit_benchmark/src/models/reward/gpt_reward_scorer.py \
        --input-dir "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_22_OmniGen2" \
        --benchmark-json "/data2/yixuan/Benchmark/version_4_200_pair_human_review/version_4_simplified_v2_translated_with_images.json" \
        --config-yaml "/data2/yixuan/image_edit_benchmark/config_iterative_refinement.yaml" \
        --output "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_22_OmniGen2/gpt_reward_results.json" \
        --api-key "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" \
        --model "gpt-4.1-2025-04-14" \
        --log-file "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_22_OmniGen2/gpt_reward_results.log"  # 可选，不指定会自动生成

python /data2/yixuan/image_edit_benchmark/src/models/reward/gpt_reward_scorer.py \
        --input-dir "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_28_hidream_e1_1" \
        --benchmark-json "/data2/yixuan/Benchmark/version_4_200_pair_human_review/version_4_simplified_v2_translated_with_images.json" \
        --config-yaml "/data2/yixuan/image_edit_benchmark/config_iterative_refinement.yaml" \
        --output "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_28_hidream_e1_1/gpt_reward_results.json" \
        --api-key "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" \
        --model "gpt-4.1-2025-04-14" \
        --log-file "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_28_hidream_e1_1/gpt_reward_results.log"  # 可选，不指定会自动生成

python /data2/yixuan/image_edit_benchmark/src/models/reward/gpt_reward_scorer.py \
        --input-dir "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_27_Ovis-U1-3B" \
        --benchmark-json "/data2/yixuan/Benchmark/version_4_200_pair_human_review/version_4_simplified_v2_translated_with_images.json" \
        --config-yaml "/data2/yixuan/image_edit_benchmark/config_iterative_refinement.yaml" \
        --output "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_27_Ovis-U1-3B/gpt_reward_results.json" \
        --api-key "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" \
        --model "gpt-4.1-2025-04-14" \
        --log-file "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_27_Ovis-U1-3B/gpt_reward_results.log"  # 可选，不指定会自动生成

python /data2/yixuan/image_edit_benchmark/src/models/reward/gpt_reward_scorer.py \
        --input-dir "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_30_Janus_4o_7B" \
        --benchmark-json "/data2/yixuan/Benchmark/version_4_200_pair_human_review/version_4_simplified_v2_translated_with_images.json" \
        --config-yaml "/data2/yixuan/image_edit_benchmark/config_iterative_refinement.yaml" \
        --output "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_30_Janus_4o_7B/gpt_reward_results.json" \
        --api-key "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" \
        --model "gpt-4.1-2025-04-14" \
        --log-file "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_30_Janus_4o_7B/gpt_reward_results.log"  # 可选，不指定会自动生成

python /data2/yixuan/image_edit_benchmark/src/models/reward/gpt_reward_scorer.py \
        --input-dir "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_18_flux_2" \
        --benchmark-json "/data2/yixuan/Benchmark/version_4_200_pair_human_review/version_4_simplified_v2_translated_with_images.json" \
        --config-yaml "/data2/yixuan/image_edit_benchmark/config_iterative_refinement.yaml" \
        --output "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_18_flux_2/gpt_reward_results.json" \
        --api-key "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" \
        --model "gpt-4.1-2025-04-14" \
        --log-file "/data2/yixuan/image_edit_benchmark/results_iterative/ICML_12_18_flux_2/gpt_reward_results.log"  # 可选，不指定会自动生成

"""

import argparse
import json
import base64
import os
import re
import yaml
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import openai
from datetime import datetime


class GPTRewardScorer:
    """使用GPT API进行图像编辑评分"""
    
    # 支持的类别（英文subset，与config.data.categories一致）
    CATEGORIES = ["physical", "environmental", "cultural", "causal", "referential"]
    
    def __init__(self, api_key: str, model: str = "gpt-4o", max_image_size: int = 1024, log_file: Optional[str] = None):
        """
        初始化GPT Reward Scorer
        
        Args:
            api_key: OpenAI API key
            model: OpenAI模型名称（如 gpt-4o, gpt-4o-mini等）
            max_image_size: 图像最大尺寸（用于缩放）
            log_file: 日志文件路径（可选）
        """
        openai.api_key = api_key
        self.model = model
        self.max_image_size = max_image_size
        self.prompts = {}  # 存储每个类别的prompt配置
        self.log_file = log_file
        self.log_fp = None
        if log_file:
            # 创建日志文件目录
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # 打开日志文件（追加模式）
            self.log_fp = open(log_file, 'a', encoding='utf-8')
    
    def _print_log(self, message: str, flush: bool = True):
        """
        同时输出到stdout和日志文件
        
        Args:
            message: 要输出的消息
            flush: 是否立即刷新
        """
        # 输出到stdout
        print(message, flush=flush)
        # 同时写入日志文件
        if self.log_fp:
            self.log_fp.write(message + '\n')
            if flush:
                self.log_fp.flush()
    
    def __del__(self):
        """析构函数，关闭日志文件"""
        if self.log_fp:
            self.log_fp.close()
    
    def load_prompts_from_config(self, config_path: str):
        """
        从配置文件加载prompt配置
        
        Args:
            config_path: YAML配置文件路径
        """
        self._print_log(f"[加载配置] 读取配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        prompts_config = config.get('prompts', {})
        
        for category in self.CATEGORIES:
            if category in prompts_config:
                self.prompts[category] = {
                    'system_prompt': prompts_config[category]['system_prompt'],
                    'user_prompt_template': prompts_config[category]['user_prompt_template']
                }
                self._print_log(f"[加载配置] 已加载类别 '{category}' 的prompt配置")
            else:
                self._print_log(f"[警告] 类别 '{category}' 的prompt配置未找到")
        
        self._print_log(f"[加载配置] 共加载 {len(self.prompts)} 个类别的prompt配置")
    
    def load_benchmark_data(self, benchmark_json_path: str) -> Dict[str, Dict]:
        """
        加载benchmark JSON数据，建立编号到数据的映射
        
        Args:
            benchmark_json_path: benchmark JSON文件路径
            
        Returns:
            字典：{编号: 数据记录}
        """
        self._print_log(f"[加载数据] 读取benchmark文件: {benchmark_json_path}")
        with open(benchmark_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 建立映射：索引 -> 数据记录
        # 注意：JSON文件是列表，索引从0开始，但图片编号可能是5位数字（如00025）
        benchmark_map = {}
        for idx, record in enumerate(data):
            # 使用索引作为key（格式化为5位数字）
            key = f"{idx:05d}"
            benchmark_map[key] = record
        
        self._print_log(f"[加载数据] 共加载 {len(benchmark_map)} 条记录")
        return benchmark_map
    
    def scan_image_files(self, input_dir: str) -> List[Dict[str, Any]]:
        """
        扫描输入目录，提取所有primary图片文件
        
        Args:
            input_dir: 输入目录路径
            
        Returns:
            图片文件信息列表，每个元素包含：path, category, index, difficulty
        """
        self._print_log(f"[扫描文件] 扫描目录: {input_dir}")
        image_files = []
        
        input_path = Path(input_dir)
        
        # 遍历每个类别文件夹
        for category in self.CATEGORIES:
            category_dir = input_path / category
            if not category_dir.exists():
                self._print_log(f"[警告] 类别目录不存在: {category_dir}")
                continue
            
            # 查找所有包含primary的图片文件
            pattern = f"*_primary.png"
            category_count = 0
            for img_file in category_dir.glob(pattern):
                # 解析文件名：{编号}_{类别}_{难度}_primary.png
                filename = img_file.stem  # 不含扩展名
                match = re.match(r'^(\d+)_(.+?)_(.+?)_primary$', filename)
                
                if match:
                    index_str = match.group(1)  # 编号（字符串，如"00025"）
                    category_name = match.group(2)  # 类别
                    difficulty = match.group(3)  # 难度
                    
                    # 验证类别是否匹配
                    if category_name == category:
                        image_files.append({
                            'path': str(img_file),
                            'category': category,
                            'index': index_str,  # 保持为字符串格式（如"00025"）
                            'difficulty': difficulty,
                            'filename': img_file.name
                        })
                        category_count += 1
                    else:
                        self._print_log(f"[警告] 文件名类别不匹配: {filename}, 目录类别: {category}")
                else:
                    self._print_log(f"[警告] 无法解析文件名: {filename}")
            
            self._print_log(f"[扫描文件] 类别 '{category}' 找到 {category_count} 个primary图片文件")
        
        self._print_log(f"[扫描文件] 共找到 {len(image_files)} 个primary图片文件")
        return image_files
    
    def _resize_image_base64(self, b64_string: str, max_size: int = None) -> str:
        """
        将base64编码的图像等比例缩放到指定最大尺寸
        
        Args:
            b64_string: base64编码的图像字符串
            max_size: 最长边的最大像素值，默认使用self.max_image_size
            
        Returns:
            缩放后的base64编码图像字符串
        """
        if max_size is None:
            max_size = self.max_image_size
        
        try:
            # 解码base64字符串
            img_data = base64.b64decode(b64_string)
            
            # 打开图像
            img = Image.open(BytesIO(img_data))
            original_size = img.size
            
            # 检查是否需要缩放
            max_dimension = max(img.size)
            if max_dimension <= max_size:
                # 不需要缩放
                return b64_string
            
            # 计算等比例缩放后的尺寸
            scale_ratio = max_size / max_dimension
            new_width = int(img.size[0] * scale_ratio)
            new_height = int(img.size[1] * scale_ratio)
            
            # 缩放图像（使用高质量的 LANCZOS 算法）
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 将缩放后的图像编码为base64
            buffer = BytesIO()
            # 保持原始格式，如果无法确定则使用JPEG
            img_format = img.format if img.format else 'JPEG'
            img_resized.save(buffer, format=img_format, quality=95)
            img_resized_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_resized_b64
            
        except Exception as e:
            self._print_log(f"[图像处理] 缩放失败: {str(e)}, 使用原始图像")
            return b64_string
    
    def _image_file_to_base64(self, image_path: str) -> str:
        """
        将本地图片文件转换为base64编码
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            base64编码的字符串
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            self._print_log(f"[错误] 读取图片文件失败: {image_path}, 错误: {str(e)}")
            raise
    
    def _extract_score_and_reasoning(self, response: str) -> Dict[str, Any]:
        """
        从GPT API响应中提取score和reasoning
        
        Args:
            response: GPT API返回的响应文本
            
        Returns:
            包含score和reasoning的字典
        """
        response_stripped = response.strip()
        
        # 方法1: 尝试解析完整JSON
        try:
            json_data = json.loads(response_stripped)
            if 'score' in json_data and 'reasoning' in json_data:
                return {
                    'score': json_data['score'],
                    'reasoning': json_data.get('reasoning', '')
                }
        except json.JSONDecodeError:
            pass
        
        # 方法2: 尝试提取JSON片段
        json_match = re.search(r'\{[^}]*"score"[^}]*"reasoning"[^}]*\}', response_stripped, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
                if 'score' in json_data:
                    return {
                        'score': json_data['score'],
                        'reasoning': json_data.get('reasoning', '')
                    }
            except:
                pass
        
        # 方法3: 尝试提取独立的JSON块
        json_blocks = re.findall(r'\{[^{}]*"score"[^{}]*\}', response_stripped, re.DOTALL)
        for block in json_blocks:
            try:
                json_data = json.loads(block)
                if 'score' in json_data:
                    return {
                        'score': json_data['score'],
                        'reasoning': json_data.get('reasoning', '')
                    }
            except:
                continue
        
        # 如果所有方法都失败，返回默认值
        self._print_log(f"[警告] 无法解析响应，返回默认值。响应内容: {response[:200]}")
        return {
            'score': 'no',
            'reasoning': f'Failed to parse response: {response[:100]}'
        }
    
    def score_single(
        self,
        edited_image_path: str,
        original_image_b64: str,
        edit_instruction: str,
        rationale: str,
        category: str,
        original_description: str = ""
    ) -> Dict[str, Any]:
        """
        对单张图像进行评分
        
        Args:
            edited_image_path: 编辑后图像的本地文件路径
            original_image_b64: 原始图像的base64编码
            edit_instruction: 编辑指令（英文）
            rationale: 任务难点描述（英文）
            category: 类别（物理、环境、社会、因果、指代）
            original_description: 原始图像描述（可选，用于user_prompt）
            
        Returns:
            评分结果字典，包含score和reasoning
        """
        # 检查类别是否有效
        if category not in self.prompts:
            raise ValueError(f"类别 '{category}' 的prompt配置未加载")
        
        prompt_config = self.prompts[category]
        system_prompt = prompt_config['system_prompt']
        user_prompt_template = prompt_config['user_prompt_template']
        
        # 读取编辑后图像并转换为base64
        edited_image_b64 = self._image_file_to_base64(edited_image_path)
        
        # 缩放图像（如果太大）
        edited_image_b64 = self._resize_image_base64(edited_image_b64)
        original_image_b64 = self._resize_image_base64(original_image_b64)
        
        # 构建user_prompt
        # 如果有rationale，需要添加到user_prompt中（根据prompt模板）
        user_prompt = user_prompt_template.format(
            original_description=original_description or "",
            edit_instruction=edit_instruction
        )
        
        # 如果有rationale，添加到user_prompt开头
        if rationale:
            user_prompt = f"Task Challenge: {rationale}\n\n{user_prompt}"
        
        # 在user_prompt末尾添加图片顺序说明，确保GPT明确理解
        user_prompt += "\n\n[Important: You will receive two images in the following order:\n1. First image: Original Image (the image before editing)\n2. Second image: Edited Image (the image after editing that needs evaluation)]"
        
        # 构造消息（参考omniedit_cot_generator_original.py的方式）
        # 在图片之间插入说明文本，明确标识每张图片
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "text", "text": "\n\n[Image 1: Original Image - This is the original image before editing]"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_image_b64}"}},
                    {"type": "text", "text": "\n[Image 2: Edited Image - This is the edited image that needs to be evaluated]"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edited_image_b64}"}}
                ],
            },
        ]
        
        # 调用GPT API
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            content = response.choices[0].message.content
            
            # 提取score和reasoning
            result = self._extract_score_and_reasoning(content)
            return result
            
        except Exception as e:
            self._print_log(f"[API错误] 调用失败: {str(e)}")
            return {
                'score': 'no',
                'reasoning': f'API Error: {str(e)}'
            }
    
    def process_all_images(
        self,
        input_dir: str,
        benchmark_json_path: str,
        config_yaml_path: str,
        output_path: str,
        batch_delay: float = 0.0
    ):
        """
        处理所有图片并生成评分结果
        
        Args:
            input_dir: 输入目录（包含5个类别文件夹）
            benchmark_json_path: benchmark JSON文件路径
            config_yaml_path: 配置文件路径
            output_path: 输出JSON文件路径
            batch_delay: 批次之间的延迟（秒），用于避免API限流
        """
        # 加载配置和数据
        self.load_prompts_from_config(config_yaml_path)
        benchmark_map = self.load_benchmark_data(benchmark_json_path)
        
        # 扫描图片文件
        image_files = self.scan_image_files(input_dir)
        
        if not image_files:
            self._print_log("[错误] 未找到任何图片文件")
            return
        
        # 按类别分组处理
        results_by_category = {cat: [] for cat in self.CATEGORIES}
        all_results = []
        
        self._print_log(f"\n[开始评分] 共 {len(image_files)} 个图片需要评分\n")
        
        # 处理每个图片
        for img_idx, img_info in enumerate(tqdm(image_files, desc="评分进度"), 1):
            category = img_info['category']
            index = img_info['index']
            edited_image_path = img_info['path']
            difficulty = img_info['difficulty']
            
            self._print_log(f"\n{'='*80}")
            self._print_log(f"[处理进度] [{img_idx}/{len(image_files)}] 处理图片: {img_info['filename']}")
            self._print_log(f"[数据提取] 编号: {index}, 类别: {category}, 难度: {difficulty}")
            
            # 从benchmark数据中获取对应记录
            if index not in benchmark_map:
                self._print_log(f"[警告] 编号 {index} 在benchmark数据中未找到，跳过")
                continue
            
            record = benchmark_map[index]
            self._print_log(f"[数据提取] 成功从benchmark数据中获取编号 {index} 的记录")
            
            # 验证类别是否匹配
            benchmark_category = record.get('subset', '')
            if benchmark_category != category:
                self._print_log(f"[警告] 编号 {index} 的类别不匹配: benchmark={benchmark_category}, 文件={category}")
                # 继续处理，但记录警告
            
            # 获取必要字段（英文版本）
            self._print_log(f"[数据提取] 开始提取必要字段（英文版本）...")
            original_image_b64 = record.get('src_img_b64', '')
            edit_instruction = record.get('edit_instruction_en', '')
            rationale = record.get('rationale_short_en', '')
            original_description = record.get('original_description_en', '')
            
            # 详细检查每个字段
            if not original_image_b64:
                self._print_log(f"[警告] 编号 {index} 缺少原始图像数据 (src_img_b64)，跳过")
                continue
            else:
                self._print_log(f"[数据提取] ✓ 原始图像数据存在 (长度: {len(original_image_b64)} 字符)")
            
            if not edit_instruction:
                self._print_log(f"[警告] 编号 {index} 缺少编辑指令 (edit_instruction_en)，跳过")
                continue
            else:
                self._print_log(f"[数据提取] ✓ 编辑指令存在: {edit_instruction[:100]}..." if len(edit_instruction) > 100 else f"[数据提取] ✓ 编辑指令存在: {edit_instruction}")
            
            if not original_description:
                self._print_log(f"[警告] 编号 {index} 缺少原始图像描述 (original_description_en)，将使用空字符串")
            else:
                self._print_log(f"[数据提取] ✓ 原始图像描述存在: {original_description[:100]}..." if len(original_description) > 100 else f"[数据提取] ✓ 原始图像描述存在: {original_description}")
            
            if not rationale:
                self._print_log(f"[警告] 编号 {index} 缺少任务难点描述 (rationale_short_en)，将使用空字符串")
            else:
                self._print_log(f"[数据提取] ✓ 任务难点描述存在: {rationale[:100]}..." if len(rationale) > 100 else f"[数据提取] ✓ 任务难点描述存在: {rationale}")
            
            self._print_log(f"[数据提取] 所有必要字段检查完成，开始调用评分API...")
            
            # 调用评分
            try:
                self._print_log(f"[API调用] 开始调用GPT API进行评分...")
                result = self.score_single(
                    edited_image_path=edited_image_path,
                    original_image_b64=original_image_b64,
                    edit_instruction=edit_instruction,
                    rationale=rationale,
                    category=category,
                    original_description=original_description
                )
                
                self._print_log(f"[API调用] 评分完成，结果: score={result['score']}, reasoning长度={len(result.get('reasoning', ''))}")
                
                # 记录结果
                sample_result = {
                    'index': index,
                    'category': category,
                    'difficulty': difficulty,
                    'score': result['score'],
                    'reasoning': result['reasoning'],
                    'edit_instruction': edit_instruction,
                    'rationale': rationale
                }
                
                results_by_category[category].append(sample_result)
                all_results.append(sample_result)
                
                self._print_log(f"[处理完成] 编号 {index} 处理成功，已添加到结果列表")
                self._print_log(f"{'='*80}\n")
                
            except Exception as e:
                self._print_log(f"[错误] 处理图片失败: {img_info['filename']}, 错误: {str(e)}")
                import traceback
                self._print_log(f"[错误详情] {traceback.format_exc()}")
                continue
            
            # 延迟（如果需要）
            if batch_delay > 0:
                import time
                time.sleep(batch_delay)
        
        # 计算统计信息
        statistics = self._calculate_statistics(results_by_category)
        
        # 生成输出JSON
        output_data = {
            'metadata': {
                'pipeline_mode': 'gpt_reward_scoring',
                'model': self.model,
                'input_dir': input_dir,
                'benchmark_json': benchmark_json_path,
                'config_yaml': config_yaml_path,
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(all_results),
                'categories': self.CATEGORIES
            },
            'category_statistics': statistics,
            'samples': all_results
        }
        
        # 保存结果
        self._print_log(f"\n[保存结果] 保存到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        self._print_log(f"[完成] 共处理 {len(all_results)} 个样本")
        
        # 打印统计摘要
        self._print_log(f"\n[统计摘要]")
        for category in self.CATEGORIES:
            stats = statistics.get(category, {})
            total = stats.get('total_count', 0)
            yes_count = stats.get('primary_yes_count', 0)
            yes_rate = stats.get('primary_yes_rate', 0.0)
            self._print_log(f"  类别 '{category}': 总数={total}, Yes={yes_count}, Yes率={yes_rate:.2f}%")
        
        # 关闭日志文件
        if self.log_fp:
            self.log_fp.close()
            self.log_fp = None
            self._print_log(f"[日志] 日志文件已关闭: {self.log_file}")
    
    def _calculate_statistics(self, results_by_category: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        计算每个类别的统计信息
        
        Args:
            results_by_category: 按类别分组的结果列表
            
        Returns:
            统计信息字典
        """
        statistics = {}
        
        for category in self.CATEGORIES:
            results = results_by_category[category]
            if not results:
                statistics[category] = {
                    'total_count': 0,
                    'primary_yes_count': 0,
                    'primary_no_count': 0,
                    'primary_yes_rate': 0.0
                }
                continue
            
            # 统计yes/no
            yes_count = sum(1 for r in results if r['score'] == 'yes' or r['score'] == 1)
            no_count = sum(1 for r in results if r['score'] == 'no' or r['score'] == 0)
            total_count = len(results)
            yes_rate = (yes_count / total_count * 100) if total_count > 0 else 0.0
            
            statistics[category] = {
                'total_count': total_count,
                'primary_yes_count': yes_count,
                'primary_no_count': no_count,
                'primary_yes_rate': yes_rate
            }
        
        return statistics


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用OpenAI GPT API作为Reward Model进行图像编辑评分"
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='输入目录路径（包含5个类别文件夹）'
    )
    parser.add_argument(
        '--benchmark-json',
        required=True,
        help='Benchmark JSON文件路径'
    )
    parser.add_argument(
        '--config-yaml',
        required=True,
        help='配置文件路径（YAML格式）'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='输出JSON文件路径'
    )
    parser.add_argument(
        '--api-key',
        required=True,
        help='OpenAI API key'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o',
        help='OpenAI模型名称（默认: gpt-4o）'
    )
    parser.add_argument(
        '--max-image-size',
        type=int,
        default=1024,
        help='图像最大尺寸（默认: 1024）'
    )
    parser.add_argument(
        '--batch-delay',
        type=float,
        default=0.0,
        help='批次之间的延迟（秒），用于避免API限流（默认: 0.0）'
    )
    parser.add_argument(
        '--log-file',
        default=None,
        help='日志文件路径（可选，如果指定则会将所有输出同时写入日志文件）'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 如果没有指定日志文件，自动生成一个（基于输出文件路径）
    log_file = args.log_file
    if not log_file:
        output_path = Path(args.output)
        log_file = str(output_path.parent / f"{output_path.stem}.log")
    
    # 创建scorer
    scorer = GPTRewardScorer(
        api_key=args.api_key,
        model=args.model,
        max_image_size=args.max_image_size,
        log_file=log_file
    )
    
    scorer._print_log(f"[初始化] GPT Reward Scorer 初始化完成")
    scorer._print_log(f"[初始化] 模型: {args.model}")
    scorer._print_log(f"[初始化] 输入目录: {args.input_dir}")
    scorer._print_log(f"[初始化] Benchmark JSON: {args.benchmark_json}")
    scorer._print_log(f"[初始化] 配置文件: {args.config_yaml}")
    scorer._print_log(f"[初始化] 输出文件: {args.output}")
    scorer._print_log(f"[初始化] 日志文件: {log_file}")
    scorer._print_log(f"[初始化] 最大图像尺寸: {args.max_image_size}")
    scorer._print_log(f"[初始化] 批次延迟: {args.batch_delay}秒")
    scorer._print_log(f"\n{'='*80}\n")
    
    # 处理所有图片
    try:
        scorer.process_all_images(
            input_dir=args.input_dir,
            benchmark_json_path=args.benchmark_json,
            config_yaml_path=args.config_yaml,
            output_path=args.output,
            batch_delay=args.batch_delay
        )
        
        scorer._print_log("\n[完成] 所有处理已完成！")
    except Exception as e:
        scorer._print_log(f"\n[致命错误] 处理过程中发生错误: {str(e)}")
        import traceback
        scorer._print_log(f"[错误详情] {traceback.format_exc()}")
        raise


if __name__ == '__main__':
    main()

