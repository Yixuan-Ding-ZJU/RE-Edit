"""
Qwen2.5-VL-7B MLLM Implementation
Qwen2.5-VL-7B MLLM实现
"""

import re
import json
import torch
from typing import Dict, List, Optional
from PIL import Image
import logging

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from ..base_mllm import BaseMllm


class Qwen25VLMllm(BaseMllm):
    """
    Qwen2.5-VL-7B MLLM实现
    
    用于分析图像编辑结果并生成改进指令
    """
    
    def __init__(self, config: dict):
        """
        初始化Qwen2.5-VL-7B模型
        
        Args:
            config: 配置字典
        """
        # 先设置所有属性（在调用super().__init__之前）
        # 因为BaseModel.__init__会立即调用_initialize()
        self.model_name = config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.device = config.get("device", "auto")
        self.dtype = config.get("dtype", "bfloat16")
        self.batch_size = config.get("batch_size", 4)
        self.max_new_tokens = config.get("max_new_tokens", 512)
        
        # 可选：flash attention配置
        self.use_flash_attention = config.get("use_flash_attention", False)
        
        # 可选：图像分辨率配置
        self.min_pixels = config.get("min_pixels", None)  # 默认 256*28*28
        self.max_pixels = config.get("max_pixels", None)  # 默认 1280*28*28
        
        # Prompt模板（针对五大类别）
        self.prompts = config.get("prompts", {})
        
        # 模型和处理器（延迟加载）
        self.model = None
        self.processor = None
        
        # 调用父类初始化（会调用_initialize）
        super().__init__(config)
        
        self.logger.info(f"Initialized Qwen2.5-VL MLLM")
        self.logger.info(f"  Model: {self.model_name}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Dtype: {self.dtype}")
        self.logger.info(f"  Flash Attention: {self.use_flash_attention}")
    
    def _initialize(self):
        """
        初始化方法（BaseModel要求）
        
        对于MLLM，我们使用延迟加载策略，所以这里不做任何事
        真正的模型加载在第一次调用analyze时进行
        """
        # 延迟加载：模型在第一次使用时才加载
        pass
    
    def _load_model(self):
        """
        实际加载Qwen2.5-VL模型
        
        基于官方代码实现
        """
        if self.model is not None:
            return  # 已经加载过了
        
        self.logger.info(f"Loading Qwen2.5-VL model from {self.model_name}...")
        
        # 准备dtype参数
        if self.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.dtype == "float16":
            torch_dtype = torch.float16
        elif self.dtype == "auto":
            torch_dtype = "auto"
        else:
            torch_dtype = "auto"
        
        # 加载模型
        if self.use_flash_attention:
            self.logger.info("Using flash_attention_2 for better performance")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map=self.device
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device
            )
        
        # 加载处理器
        if self.min_pixels is not None and self.max_pixels is not None:
            self.logger.info(f"Using custom pixel range: {self.min_pixels} - {self.max_pixels}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels
            )
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # 为decoder-only模型设置left padding（避免warning）
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = 'left'
            self.logger.info("Set tokenizer padding_side to 'left' for decoder-only architecture")
        
        self.logger.info("Qwen2.5-VL model loaded successfully")
        self.logger.info(f"  Model device: {self.model.device if hasattr(self.model, 'device') else 'distributed'}")
    
    def analyze(self,
                edited_image: Image.Image,
                original_image: Image.Image,
                original_description: str,
                edit_instruction: str,
                category: str,
                **kwargs) -> Dict[str, str]:
        """
        分析单个图像编辑结果
        
        Args:
            edited_image: 编辑后的图像
            original_image: 原始图像
            original_description: 原图描述
            edit_instruction: 编辑指令
            category: 类别名称
            **kwargs: 其他参数
        
        Returns:
            {
                "cot": "推理过程...",
                "re_edit_instruction": "改进指令...",
                "raw_output": "原始输出"
            }
        """
        if self.model is None:
            self._load_model()
        
        # 构建prompt
        system_prompt = self._get_system_prompt(category)
        user_prompt = self._build_user_prompt(
            original_description, 
            edit_instruction, 
            category
        )
        
        # 构建messages格式
        # 注意：Qwen2.5-VL的格式与Qwen3-VL不同，system可以是纯文本
        messages = [
            {
                "role": "system",
                "content": system_prompt  # Qwen2.5-VL支持纯文本system
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},   # 原图
                    {"type": "image", "image": edited_image},     # 编辑后的图
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        try:
            # 调用模型生成
            raw_output = self._generate_single(messages)
        except Exception as e:
            self.logger.error(f"MLLM generation failed: {e}")
            raw_output = ""
        
        # 解析输出
        parsed_output = self.parse_output(raw_output)
        
        # 验证输出
        if not self.validate_output(parsed_output):
            self.logger.warning(f"Invalid MLLM output, using fallback instruction")
            parsed_output["re_edit_instruction"] = self.get_fallback_instruction(
                edit_instruction, category
            )
        
        return parsed_output
    
    def _generate_single(self, messages: List[Dict]) -> str:
        """
        生成单个响应
        
        Args:
            messages: 消息列表
        
        Returns:
            生成的文本
        """
        # 准备输入（基于官方代码）
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 移动到正确的设备
        if hasattr(self.model, 'device'):
            inputs = inputs.to(self.model.device)
        else:
            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # 生成
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        
        # 解码（只返回生成的新token）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""
    
    def batch_analyze(self,
                      edited_images: List[Image.Image],
                      original_images: List[Image.Image],
                      original_descriptions: List[str],
                      edit_instructions: List[str],
                      categories: List[str],
                      batch_size: int = 4,
                      on_batch_complete: callable = None,
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
            on_batch_complete: 可选回调函数，在每个batch完成后调用
                               签名: on_batch_complete(batch_idx, batch_results, batch_indices)
            **kwargs: 其他参数
        
        Returns:
            结果列表
        """
        if self.model is None:
            self._load_model()
        
        n = len(edited_images)
        results = []
        
        # 使用配置的batch_size
        actual_batch_size = batch_size if batch_size > 0 else self.batch_size
        
        self.logger.info(f"Batch analyzing {n} images with batch_size={actual_batch_size}")
        
        # 批量推理
        from tqdm import tqdm
        
        batch_idx = 0
        for batch_start in tqdm(range(0, n, actual_batch_size), desc="MLLM Analyzing"):
            batch_end = min(batch_start + actual_batch_size, n)
            
            # 构建当前batch的messages
            batch_messages = []
            for i in range(batch_start, batch_end):
                # 验证输入数据
                if original_images[i] is None:
                    self.logger.error(f"original_image is None for sample {i} (global index: {batch_start + i})")
                    raise ValueError(f"original_image is None for sample {i}")
                if edited_images[i] is None:
                    self.logger.error(f"edited_image is None for sample {i} (global index: {batch_start + i})")
                    raise ValueError(f"edited_image is None for sample {i}")
                
                system_prompt = self._get_system_prompt(categories[i])
                user_prompt = self._build_user_prompt(
                    original_descriptions[i] if original_descriptions[i] is not None else "",
                    edit_instructions[i] if edit_instructions[i] is not None else "",
                    categories[i] if categories[i] is not None else "unknown"
                )
                
                # 确保 user_prompt 不为 None 或空
                if not user_prompt or not isinstance(user_prompt, str):
                    self.logger.warning(f"user_prompt is invalid for sample {i}, using default")
                    user_prompt = f"<desired_editing_instruction>{edit_instructions[i] if edit_instructions[i] else ''}</desired_editing_instruction>\nReturn reasoning and one re-editing instruction as specified."
                
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": original_images[i]},
                            {"type": "image", "image": edited_images[i]},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
                batch_messages.append(messages)
            
            # 批量生成
            try:
                batch_outputs = self._generate_batch(batch_messages)
            except Exception as e:
                self.logger.error(f"Batch generation failed: {e}, falling back to sequential")
                # Fallback: sequential processing
                batch_outputs = []
                for messages in batch_messages:
                    try:
                        output = self._generate_single(messages)
                        batch_outputs.append(output)
                    except Exception as e2:
                        self.logger.error(f"Sequential generation also failed: {e2}")
                        batch_outputs.append("")
            
            # 解析和验证输出
            batch_results = []
            batch_indices = []
            for i, raw_output in enumerate(batch_outputs):
                global_idx = batch_start + i
                parsed_output = self.parse_output(raw_output)
                
                if not self.validate_output(parsed_output):
                    self.logger.warning(f"Invalid MLLM output for sample {global_idx}, using fallback")
                    parsed_output["re_edit_instruction"] = self.get_fallback_instruction(
                        edit_instructions[global_idx], categories[global_idx]
                    )
                
                results.append(parsed_output)
                batch_results.append(parsed_output)
                batch_indices.append(global_idx)
            
            # 调用回调函数实现实时输出
            if on_batch_complete is not None:
                try:
                    on_batch_complete(batch_idx, batch_results, batch_indices)
                except Exception as e:
                    self.logger.error(f"Error in batch completion callback: {e}")
            
            batch_idx += 1
        
        self.logger.info(f"Batch analysis completed for {n} images")
        return results
    
    def _generate_batch(self, batch_messages: List[List[Dict]]) -> List[str]:
        """
        批量生成响应
        
        Args:
            batch_messages: 批量消息列表
        
        Returns:
            生成的文本列表
        """
        # 准备批量输入
        texts = []
        all_image_inputs = []
        all_video_inputs = []
        
        for messages in batch_messages:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            
            image_inputs, video_inputs = process_vision_info(messages)
            all_image_inputs.append(image_inputs)
            all_video_inputs.append(video_inputs)
        
        # 合并图像输入（flatten）
        merged_images = []
        for img_list in all_image_inputs:
            if img_list:
                merged_images.extend(img_list)
        
        # 处理输入
        inputs = self.processor(
            text=texts,
            images=merged_images if merged_images else None,
            videos=None,  # 通常不使用视频
            padding=True,
            return_tensors="pt",
        )
        
        # 移动到设备
        if hasattr(self.model, 'device'):
            inputs = inputs.to(self.model.device)
        else:
            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # 批量生成
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        
        # 批量解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_texts
    
    def _get_system_prompt(self, category: str) -> str:
        """
        获取类别对应的系统prompt
        
        Args:
            category: 类别名称
        
        Returns:
            系统prompt
        """
        # 处理 None 值
        if category is None:
            category = "unknown"
        category = str(category) if category else "unknown"
        
        category_config = self.prompts.get(category, {})
        system_prompt = category_config.get("system", self._get_default_system_prompt())
        
        # 确保返回的是非空字符串
        if not system_prompt or not isinstance(system_prompt, str):
            system_prompt = self._get_default_system_prompt()
        
        return system_prompt
    
    def _get_default_system_prompt(self) -> str:
        """
        获取默认系统prompt
        
        Returns:
            默认系统prompt
        """
        return """You are an expert in analyzing image editing quality.
Your task is to:
1. Analyze the editing result by comparing the original and edited images
2. Identify any issues or areas for improvement
3. Provide a refined editing instruction to improve the result

Please structure your response as:
## Chain of Thought
[Your reasoning process here]

## Re-edit Instruction
[Your improved editing instruction here]
"""
    
    def _build_user_prompt(self,
                           original_description: str,
                           edit_instruction: str,
                           category: str) -> str:
        """
        构建用户prompt
        
        Args:
            original_description: 原图描述
            edit_instruction: 编辑指令
            category: 类别名称
        
        Returns:
            用户prompt
        """
        # 处理 None 值，提供默认值
        if original_description is None:
            original_description = ""
        if edit_instruction is None:
            edit_instruction = ""
        if category is None:
            category = "unknown"
        
        # 确保都是字符串类型
        original_description = str(original_description) if original_description else ""
        edit_instruction = str(edit_instruction) if edit_instruction else ""
        category = str(category) if category else "unknown"
        
        category_config = self.prompts.get(category, {})
        template = category_config.get("user_template", self._get_default_user_template())
        
        # 替换模板变量
        try:
            user_prompt = template.format(
                original_description=original_description,
                edit_instruction=edit_instruction,
                category=category
            )
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Error formatting user prompt template: {e}, using default template")
            # 使用默认模板作为 fallback
            user_prompt = self._get_default_user_template().format(
                original_description=original_description,
                edit_instruction=edit_instruction,
                category=category
            )
        
        # 确保返回的是非空字符串
        if not user_prompt or not user_prompt.strip():
            user_prompt = f"<desired_editing_instruction>{edit_instruction}</desired_editing_instruction>\nReturn reasoning and one re-editing instruction as specified."
        
        return user_prompt
    
    def _get_default_user_template(self) -> str:
        """
        获取默认用户prompt模板
        
        Returns:
            默认用户prompt模板
        """
        return """<desired_editing_instruction>{edit_instruction}</desired_editing_instruction>
Return reasoning and one re-editing instruction as specified.
"""
    
    def parse_output(self, raw_output: str) -> Dict[str, str]:
        """
        解析MLLM输出，提取CoT和Re-edit指令
        
        支持格式（优先级从高到低）：
        1. XML标签格式: <CoT>...</CoT> <Re_edit>...</Re_edit> （推荐，与prompt匹配）
        2. JSON格式: {"cot": "...", "re_edit_instruction": "..."}
        3. Markdown格式: ## Chain of Thought ... ## Re-edit Instruction ...
        4. 纯文本格式（使用智能fallback）
        
        Args:
            raw_output: MLLM原始输出
        
        Returns:
            解析后的字典 {"cot": str, "re_edit_instruction": str, "raw_output": str}
        """
        # ===== 方法1: XML标签格式 (优先，与prompt设计匹配) =====
        # 支持单行和多行格式
        # 单行: <CoT>content</CoT>
        # 多行: <Re_edit>\n  content\n</Re_edit>
        
        # 提取CoT（可能有多个）
        cot_pattern = r'<CoT[^>]*>\s*(.*?)\s*</CoT>'
        cot_matches = re.findall(cot_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        
        # 提取Re_edit（可能有多个，与EasyR1框架保持一致）
        re_edit_pattern = r'<Re_edit[^>]*>\s*(.*?)\s*</Re_edit>'
        re_edit_matches = re.findall(re_edit_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        
        if cot_matches or re_edit_matches:
            # 合并所有CoT块（使用换行符连接）
            cot_text = "\n".join([self._clean_text(cot) for cot in cot_matches if cot.strip()])
            
            # 合并所有Re_edit指令（使用分号+空格连接，与EasyR1框架一致）
            if re_edit_matches:
                re_edit_texts = [self._clean_text(re) for re in re_edit_matches if re.strip()]
                re_edit_text = "; ".join(re_edit_texts)  # 使用 "; " 拼接多个指令
                if self._validate_instruction(re_edit_text):
                    return {
                        "cot": cot_text,
                        "re_edit_instruction": re_edit_text,  # 拼接后的字符串（向后兼容）
                        "re_edit_instructions": re_edit_texts,  # 独立的指令列表（用于迭代refinement）
                        "raw_output": raw_output
                    }
        
        # ===== 方法2: JSON格式 =====
        json_result = self._extract_from_json(raw_output)
        if json_result:
            return json_result
        
        # ===== 方法3: Markdown格式 =====
        markdown_result = self._extract_from_markdown(raw_output)
        if markdown_result:
            return markdown_result
        
        # ===== 方法4: 智能Fallback =====
        # 尝试从文本中提取指令模式
        fallback_instruction = self._extract_instruction_pattern(raw_output)
        
        self.logger.warning(f"Failed to parse MLLM output with standard formats, using fallback")
        # 只在第一次失败时输出详细信息
        if not hasattr(self, '_parse_fail_logged'):
            self._parse_fail_logged = True
            self.logger.debug(f"Raw MLLM output (first failure):\n{raw_output[:500]}...")
        
        # 对于fallback，单个指令包装为列表
        return {
            "cot": "",
            "re_edit_instruction": fallback_instruction,  # 拼接后的字符串（向后兼容）
            "re_edit_instructions": [fallback_instruction] if fallback_instruction else [],  # 独立的指令列表
            "raw_output": raw_output
        }
    
    def _clean_text(self, text: str) -> str:
        """
        清理提取的文本
        
        Args:
            text: 原始文本
        
        Returns:
            清理后的文本
        """
        # 去除首尾空白
        text = text.strip()
        
        # 去除首尾引号
        text = re.sub(r'^["\']|["\']$', '', text)
        
        # 压缩多余的空格（但保留换行）
        text = re.sub(r' +', ' ', text)
        
        # 压缩多余的换行
        text = re.sub(r'\n\n+', '\n', text)
        
        return text.strip()
    
    def _validate_instruction(self, instruction: str) -> bool:
        """
        验证指令是否有效
        
        Args:
            instruction: 待验证的指令
        
        Returns:
            True if valid, False otherwise
        """
        if not instruction or len(instruction.strip()) < 5:
            return False
        
        # 检查无效模式
        invalid_patterns = [
            r'^\s*$',           # 空白
            r'^[{}[\]]+$',      # 只有括号
            r'^["\']+$',        # 只有引号
            r'^[.,;:!?]+$',     # 只有标点
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, instruction):
                return False
        
        return True
    
    def _extract_from_json(self, text: str) -> Optional[Dict[str, str]]:
        """从JSON格式提取"""
        try:
            # 尝试解析整个文本为JSON
            data = json.loads(text.strip())
            if isinstance(data, dict):
                re_edit = data.get("re_edit_instruction") or data.get("Re_edit") or data.get("re_edit")
                if re_edit and self._validate_instruction(str(re_edit)):
                    re_edit_text = self._clean_text(str(re_edit))
                    # 对于JSON格式，如果re_edit是列表，则使用列表；否则包装为列表
                    if isinstance(re_edit, list):
                        re_edit_list = [self._clean_text(str(r)) for r in re_edit if self._validate_instruction(str(r))]
                    else:
                        re_edit_list = [re_edit_text] if re_edit_text else []
                    
                    return {
                        "cot": str(data.get("cot", "")),
                        "re_edit_instruction": re_edit_text,  # 拼接后的字符串（向后兼容）
                        "re_edit_instructions": re_edit_list,  # 独立的指令列表
                        "raw_output": text
                    }
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _extract_from_markdown(self, text: str) -> Optional[Dict[str, str]]:
        """从Markdown格式提取"""
        cot_match = re.search(r'##\s*Chain of Thought\s*\n(.*?)(?=##|$)', 
                             text, re.DOTALL | re.IGNORECASE)
        re_edit_match = re.search(r'##\s*Re-?edit Instruction\s*\n(.*?)$', 
                                  text, re.DOTALL | re.IGNORECASE)
        
        if cot_match or re_edit_match:
            cot_text = self._clean_text(cot_match.group(1)) if cot_match else ""
            re_edit_text = self._clean_text(re_edit_match.group(1)) if re_edit_match else ""
            
            if self._validate_instruction(re_edit_text):
                # 对于Markdown格式，单个指令包装为列表
                return {
                    "cot": cot_text,
                    "re_edit_instruction": re_edit_text,  # 拼接后的字符串（向后兼容）
                    "re_edit_instructions": [re_edit_text],  # 独立的指令列表
                    "raw_output": text
                }
        
        return None
    
    def _extract_instruction_pattern(self, text: str) -> str:
        """
        从文本中提取类似指令的模式（fallback）
        
        Args:
            text: 原始文本
        
        Returns:
            提取的指令或默认fallback
        """
        # 查找包含动作动词的句子
        instruction_patterns = [
            r'(?:improve|enhance|adjust|modify|edit|change|fix|correct|make|add|remove|increase|decrease|brighten|darken)\s+[^.!?\n]+[.!?]?',
        ]
        
        for pattern in instruction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # 返回最长的匹配
                instruction = max(matches, key=len).strip()
                if self._validate_instruction(instruction):
                    return self._clean_text(instruction)
        
        # 如果一切都失败，返回清理后的整个文本（限制长度）
        cleaned = self._clean_text(text)
        if len(cleaned) > 200:
            cleaned = cleaned[:200] + "..."
        
        return cleaned if self._validate_instruction(cleaned) else "Improve the image quality and consistency."
    
    def validate_output(self, output: Dict[str, str]) -> bool:
        """
        验证MLLM输出是否有效（覆盖基类方法）
        
        Args:
            output: 解析后的输出字典
        
        Returns:
            True if valid, False otherwise
        """
        re_edit = output.get("re_edit_instruction", "").strip()
        return self._validate_instruction(re_edit)
    
    def get_fallback_instruction(self, original_instruction: str, category: str) -> str:
        """
        当MLLM输出无效时，返回fallback指令（覆盖基类方法）
        
        Args:
            original_instruction: 原始编辑指令
            category: 类别名称
        
        Returns:
            fallback指令
        """
        # 策略：使用原始指令的增强版本
        # 格式：保持原指令，添加通用的质量改进提示
        return f"{original_instruction.strip()}, ensuring better quality and consistency"
    
    def __call__(self, *args, **kwargs):
        """
        模型调用接口（实现基类抽象方法）
        
        这是一个wrapper，调用analyze方法
        """
        return self.analyze(*args, **kwargs)
    
    def load_to_gpu(self):
        """将模型加载到GPU"""
        self.logger.info("[Qwen2.5-VL MLLM] Loading to GPU...")
        if self.model is None:
            self._load_model()
        # 模型已经通过device_map="auto"自动分配到GPU
    
    def unload_from_gpu(self):
        """
        从GPU卸载模型（优化版）
        
        优化策略：
        1. 检测模型分布在哪些GPU上
        2. 并行清理每个GPU的缓存
        3. 显式垃圾回收
        """
        self.logger.info("[Qwen2.5-VL MLLM] Unloading from GPU...")
        if self.model is not None:
            import torch
            
            # 检测模型分布在哪些GPU上
            device_set = set()
            if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                # device_map可能是: {"layer.0": 0, "layer.1": 1, ...}
                for device in self.model.hf_device_map.values():
                    if isinstance(device, int):
                        device_set.add(device)
                    elif isinstance(device, str) and 'cuda' in device:
                        # 处理 "cuda:0" 格式
                        try:
                            device_id = int(device.split(':')[-1])
                            device_set.add(device_id)
                        except:
                            pass
                self.logger.info(f"[Qwen2.5-VL MLLM] Model distributed across GPUs: {sorted(device_set)}")
            
            # 删除模型和处理器
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            # 建议2: 显式垃圾回收
            import gc
            gc.collect()
            self.logger.info("[Qwen2.5-VL MLLM] Garbage collection completed")
            
            # 建议1: 并行清理每个GPU的缓存
            if torch.cuda.is_available():
                if device_set:
                    # 多GPU并行清理
                    from concurrent.futures import ThreadPoolExecutor
                    
                    def clear_cache_on_device(device_id):
                        """清理指定GPU的缓存"""
                        try:
                            with torch.cuda.device(device_id):
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            return device_id, True
                        except Exception as e:
                            return device_id, False
                    
                    with ThreadPoolExecutor(max_workers=len(device_set)) as executor:
                        results = list(executor.map(clear_cache_on_device, sorted(device_set)))
                    
                    # 记录清理结果
                    success_gpus = [dev for dev, success in results if success]
                    self.logger.info(f"[Qwen2.5-VL MLLM] Cleared cache on GPUs: {success_gpus}")
                else:
                    # 单GPU或未检测到device_map，使用默认清理
                    torch.cuda.empty_cache()
                    self.logger.info("[Qwen2.5-VL MLLM] Cleared cache on default device")
            
            self.logger.info("[Qwen2.5-VL MLLM] Model unloaded successfully")

