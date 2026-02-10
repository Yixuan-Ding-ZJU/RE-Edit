"""
Qwen3-VL Reward Model Implementation
基于Qwen3-VL的图像评分模型

支持本地部署的Qwen3-VL-30B模型
"""

import re
import torch
from PIL import Image
from typing import Any, Dict, Optional
import io

from ..base_reward import BaseRewardModel


class Qwen3VLRewardModel(BaseRewardModel):
    """
    基于Qwen3-VL的Reward评分模型
    
    使用Vision-Language模型对编辑后的图像进行评分
    """
    
    def _initialize(self):
        """
        初始化Qwen3-VL模型
        """
        from transformers import AutoModelForImageTextToText, AutoProcessor
        
        # 从配置获取参数
        self.model_name = self.config.get("model_name", "Qwen/Qwen3-VL-30B-Instruct")
        self.device = self.config.get("device", "auto")
        self.dtype = self.config.get("dtype", "bfloat16")
        self.max_new_tokens = self.config.get("max_new_tokens", 128)
        self.use_flash_attention = self.config.get("use_flash_attention", False)
        
        print(f"[Qwen3VLRewardModel] 正在加载模型: {self.model_name}")
        print(f"[Qwen3VLRewardModel] 设备: {self.device}, 数据类型: {self.dtype}")
        
        # 设置数据类型
        if self.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.dtype == "float16":
            dtype = torch.float16
        elif self.dtype == "float32":
            dtype = torch.float32
        else:
            dtype = "auto"
        
        # 加载模型
        model_kwargs = {
            "dtype": dtype,
            "device_map": self.device if self.device != "cuda" else "auto",
        }
        
        # 如果使用flash attention
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("[Qwen3VLRewardModel] 启用 Flash Attention 2")
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        print("[Qwen3VLRewardModel] 模型加载完成")
        
        # 加载processor
        print("[Qwen3VLRewardModel] 正在加载 Processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        print("[Qwen3VLRewardModel] 初始化完成")
    
    def score(self,
              edited_image: Image.Image,
              original_description: str,
              edit_instruction: str,
              system_prompt: str,
              user_prompt: str,
              original_image: Optional[Image.Image] = None,
              **kwargs) -> float:
        """
        对编辑后的图像进行评分
        
        Args:
            edited_image: 编辑后的PIL图像
            original_description: 原始图像描述
            edit_instruction: 编辑指令
            system_prompt: 系统prompt
            user_prompt: 用户prompt
            original_image: 原始图像（可选，某些评分可能需要对比）
            **kwargs: 其他参数
            
        Returns:
            评分（0-10的浮点数）
        """
        # 确保图像是RGB格式
        if edited_image.mode != 'RGB':
            edited_image = edited_image.convert('RGB')
        
        # 构建messages
        # 根据是否需要原图来决定消息格式
        if original_image is not None and kwargs.get("compare_with_original", False):
            # 如果需要对比原图
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Original image:"},
                        {"type": "image", "image": original_image},
                        {"type": "text", "text": "Edited image:"},
                        {"type": "image", "image": edited_image},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
        else:
            # 只使用编辑后的图像
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": edited_image},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
        
        # 准备输入
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # 生成输出
        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        
        # 从输出文本中提取分数
        response_text = output_text[0] if output_text else ""
        score = self._extract_score_from_response(response_text)
        
        return score
    
    def _extract_score_from_response(self, response: str) -> float:
        """
        从模型响应中提取分数
        
        支持多种格式：
        1. JSON格式（新）: {"reasoning": "...", "score": 0或1}
        2. 数值格式（旧）: "Score: 8.5", "8.5/10", "Rating: 8.5", "8.5"
        3. 文本格式（旧）: "yes", "no"
        
        Args:
            response: 模型的文本响应
            
        Returns:
            提取的分数（0-10范围内，或转换后的0/10）
        """
        response_stripped = response.strip()
        
        # ===== 方法1: 尝试解析JSON格式（新格式） =====
        try:
            import json
            json_data = json.loads(response_stripped)
            
            if 'score' in json_data:
                score = json_data['score']
                
                # 处理不同的score格式
                if isinstance(score, (int, float)):
                    # score为数字：0或1（二元），或0-10（数值）
                    if score in [0, 1]:
                        # 二元评分：转换为0或10
                        return float(score * 10)
                    else:
                        # 数值评分：限制在0-10范围
                        return max(0.0, min(10.0, float(score)))
                
                elif isinstance(score, str):
                    score_lower = score.strip().lower()
                    if score_lower in ["1", "yes"]:
                        return 10.0
                    elif score_lower in ["0", "no"]:
                        return 0.0
                    else:
                        # 尝试转换为数字
                        try:
                            return max(0.0, min(10.0, float(score)))
                        except:
                            pass
                
                elif isinstance(score, list) and len(score) > 0:
                    score_val = score[0]
                    if score_val in [1, "1", "yes"]:
                        return 10.0
                    elif score_val in [0, "0", "no"]:
                        return 0.0
        
        except json.JSONDecodeError:
            # JSON解析失败，尝试提取JSON片段
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', response_stripped, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    if 'score' in json_data:
                        score = json_data['score']
                        if score in [1, "1", "yes"]:
                            return 10.0
                        elif score in [0, "0", "no"]:
                            return 0.0
                except:
                    pass
        
        except Exception:
            pass
        
        # ===== 方法2: 检查yes/no文本（向后兼容） =====
        response_lower = response_stripped.lower()
        if re.search(r'\byes\b', response_lower):
            return 10.0
        elif re.search(r'\bno\b', response_lower):
            return 0.0
        
        # ===== 方法3: 尝试提取数值分数（旧格式） =====
        patterns = [
            r'[Ss]core[:\s]+(\d+\.?\d*)',  # Score: 8.5
            r'[Rr]ating[:\s]+(\d+\.?\d*)',  # Rating: 8.5
            r'(\d+\.?\d*)\s*/\s*10',         # 8.5/10 or 8.5 / 10
            r'(\d+\.?\d*)\s*out of\s*10',   # 8.5 out of 10
            r'^(\d+\.?\d*)$',                # 单独的数字
            r'(\d+\.?\d*)',                  # 任何数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_stripped)
            if match:
                try:
                    score = float(match.group(1))
                    # 限制在0-10范围内
                    score = max(0.0, min(10.0, score))
                    return score
                except (ValueError, IndexError):
                    continue
        
        # 如果无法提取，返回中等分数并记录警告
        print(f"[Qwen3VLRewardModel] 警告: 无法从响应中提取分数: '{response[:100]}...'")
        return 5.0  # 默认中等分数
    
    def batch_score(self,
                   edited_images: list,
                   original_descriptions: list,
                   edit_instructions: list,
                   system_prompts: list,
                   user_prompts: list,
                   original_images: Optional[list] = None,
                   **kwargs) -> list:
        """
        批量评分（使用真正的batch inference）
        
        根据Qwen官方推荐，使用batch inference提升吞吐量
        
        Args:
            edited_images: 编辑后的图像列表
            original_descriptions: 原始图像描述列表
            edit_instructions: 编辑指令列表
            system_prompts: 系统prompt列表
            user_prompts: 用户prompt列表
            original_images: 原始图像列表（可选）
            **kwargs: 其他参数
                - batch_size: 批处理大小（默认4）
                - use_batch_inference: 是否使用batch inference（默认True）
            
        Returns:
            评分列表
        """
        n = len(edited_images)
        if not all(len(lst) == n for lst in [original_descriptions, edit_instructions, 
                                              system_prompts, user_prompts]):
            raise ValueError("All input lists must have the same length")
        
        if original_images is None:
            original_images = [None] * n
        
        # 检查是否使用batch inference
        use_batch = kwargs.get("use_batch_inference", True)
        if not use_batch:
            # 回退到逐个处理（向后兼容）
            return self._batch_score_sequential(
                edited_images, original_descriptions, edit_instructions,
                system_prompts, user_prompts, original_images, **kwargs
            )
        
        # 使用batch inference
        batch_size = kwargs.get("batch_size", 4)
        print(f"[Qwen3VLRewardModel] Batch scoring {n} images with batch_size={batch_size}")
        
        # 设置padding_side为left（Qwen官方推荐用于batch generation）
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = 'left'
        
        all_scores = []
        
        try:
            # 分批处理
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                batch_indices = range(batch_start, batch_end)
                
                # 构建batch messages
                batch_messages = []
                for i in batch_indices:
                    messages = self._build_messages(
                        edited_images[i],
                        system_prompts[i],
                        user_prompts[i],
                        original_images[i],
                        kwargs.get("compare_with_original", False)
                    )
                    batch_messages.append(messages)
                
                # Batch推理
                try:
                    # 准备输入（添加padding参数）
                    inputs = self.processor.apply_chat_template(
                        batch_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True  # 关键：batch inference需要padding
                    )
                    inputs = inputs.to(self.model.device)
                    
                    # 生成输出
                    max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
                    
                    with torch.inference_mode():
                        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_texts = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                    
                    # 解析分数
                    batch_scores = [self._extract_score_from_response(text) for text in output_texts]
                    all_scores.extend(batch_scores)
                    
                    print(f"[Qwen3VLRewardModel] Processed batch {batch_start}-{batch_end-1}: avg_score={sum(batch_scores)/len(batch_scores):.3f}")
                    
                except Exception as e:
                    print(f"[Qwen3VLRewardModel] Error in batch {batch_start}-{batch_end-1}: {e}")
                    print(f"[Qwen3VLRewardModel] Falling back to sequential processing for this batch...")
                    # 回退到逐个处理这个batch
                    for i in batch_indices:
                        try:
                            score = self.score(
                                edited_image=edited_images[i],
                                original_description=original_descriptions[i],
                                edit_instruction=edit_instructions[i],
                                system_prompt=system_prompts[i],
                                user_prompt=user_prompts[i],
                                original_image=original_images[i],
                                **kwargs
                            )
                            all_scores.append(score)
                        except Exception as e2:
                            print(f"[Qwen3VLRewardModel] Error scoring image {i}: {e2}")
                            all_scores.append(5.0)  # 默认分数
        
        finally:
            # 恢复原始padding_side
            self.processor.tokenizer.padding_side = original_padding_side
        
        return all_scores
    
    def _build_messages(self, edited_image, system_prompt, user_prompt, 
                       original_image=None, compare_with_original=False):
        """
        构建单个样本的messages
        
        Args:
            edited_image: 编辑后的图像
            system_prompt: 系统prompt
            user_prompt: 用户prompt
            original_image: 原始图像（可选）
            compare_with_original: 是否对比原图
            
        Returns:
            messages列表
        """
        # 确保图像是RGB格式
        if edited_image.mode != 'RGB':
            edited_image = edited_image.convert('RGB')
        
        if original_image is not None and compare_with_original:
            # 如果需要对比原图
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Original image:"},
                        {"type": "image", "image": original_image},
                        {"type": "text", "text": "Edited image:"},
                        {"type": "image", "image": edited_image},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
        else:
            # 只使用编辑后的图像
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": edited_image},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
        
        return messages
    
    def _batch_score_sequential(self,
                                edited_images: list,
                                original_descriptions: list,
                                edit_instructions: list,
                                system_prompts: list,
                                user_prompts: list,
                                original_images: Optional[list] = None,
                                **kwargs) -> list:
        """
        批量评分（逐个处理，用于向后兼容或batch inference失败时的回退）
        
        Args:
            同batch_score
            
        Returns:
            评分列表
        """
        scores = []
        n = len(edited_images)
        
        for i in range(n):
            try:
                score = self.score(
                    edited_image=edited_images[i],
                    original_description=original_descriptions[i],
                    edit_instruction=edit_instructions[i],
                    system_prompt=system_prompts[i],
                    user_prompt=user_prompts[i],
                    original_image=original_images[i] if original_images else None,
                    **kwargs
                )
                scores.append(score)
            except Exception as e:
                print(f"[Qwen3VLRewardModel] Error scoring image {i}: {e}")
                scores.append(5.0)  # 默认分数
        
        return scores
    
    def unload_from_gpu(self):
        """
        将模型从GPU卸载到CPU，释放GPU内存
        """
        if hasattr(self, 'model') and self.model is not None:
            print(f"[Qwen3VLRewardModel] 将模型从GPU卸载到CPU...")
            self.model.to('cpu')
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Qwen3VLRewardModel] 模型已卸载到CPU")
    
    def load_to_gpu(self):
        """
        将模型从CPU加载到GPU
        """
        if hasattr(self, 'model') and self.model is not None:
            print(f"[Qwen3VLRewardModel] 将模型从CPU加载到GPU...")
            # 使用初始化时的device配置
            if self.device == "cuda":
                target_device = "cuda"
            elif self.device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_device = self.device
            
            self.model.to(target_device)
            print(f"[Qwen3VLRewardModel] 模型已加载到GPU: {target_device}")
    
    def __del__(self):
        """
        清理资源
        """
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

