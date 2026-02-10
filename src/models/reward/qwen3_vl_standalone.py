#!/usr/bin/env python3
"""
Qwen3-VL Standalone评分脚本
用于在独立虚拟环境中运行Qwen3-VL模型

使用方法：
    python qwen3_vl_standalone.py --input input.json --output output.json
"""

import argparse
import json
import base64
import sys
from pathlib import Path
from io import BytesIO
from typing import List, Dict
import re

# 在新环境中导入Qwen3-VL
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    import torch
except ImportError as e:
    print(f"Error: Failed to import required packages: {e}", file=sys.stderr, flush=True)
    print("Please install: pip install transformers pillow torch", file=sys.stderr, flush=True)
    sys.exit(1)


class Qwen3VLStandaloneScorer:
    """独立的Qwen3-VL评分器"""
    
    def __init__(self, model_name: str, device: str = "auto", dtype: str = "bfloat16"):
        """
        初始化模型
        
        Args:
            model_name: 模型名称或路径
            device: 设备（auto, cuda, cpu）
            dtype: 数据类型
        """
        print(f"[Qwen3VL-Standalone] Loading model: {model_name}", file=sys.stderr, flush=True)
        
        # 解析dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = "auto"
        
        # 加载模型
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device
        )
        
        # 加载processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.device = next(self.model.parameters()).device
        print(f"[Qwen3VL-Standalone] Model loaded on device: {self.device}", file=sys.stderr, flush=True)
    
    def decode_base64_image(self, base64_str: str) -> Image.Image:
        """解码base64图像"""
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def extract_score_and_reasoning(self, response: str) -> dict:
        """
        从响应中提取score和reasoning
        
        支持多种格式：
        1. JSON格式（PQ指标）: {"reasoning": "...", "score": [naturalness, artifacts]}
        2. JSON格式（二元）: {"reasoning": "...", "score": 0或1}
        3. 纯文本格式: "yes" 或 "no" （向后兼容，reasoning为空）
        
        Returns:
            {"score": "yes"/"no" 或 [naturalness, artifacts], "reasoning": "..."}
        """
        response_stripped = response.strip()
        
        # ===== 方法1: 尝试解析JSON格式 =====
        try:
            json_data = json.loads(response_stripped)
            
            if 'score' in json_data:
                score = json_data['score']
                reasoning = json_data.get('reasoning', '')
                
                # 检查score是否为列表格式（PQ指标）
                if isinstance(score, list) and len(score) >= 2:
                    # PQ指标格式: [naturalness, artifacts]
                    try:
                        naturalness = float(score[0])
                        artifacts = float(score[1])
                        # 确保分数在0-10范围内
                        naturalness = max(0.0, min(10.0, naturalness))
                        artifacts = max(0.0, min(10.0, artifacts))
                        return {"score": [naturalness, artifacts], "reasoning": reasoning}
                    except (ValueError, IndexError, TypeError):
                        # 如果转换失败，继续尝试其他格式
                        pass
                
                # 如果不是列表，使用原有的yes/no提取逻辑
                score_str = self.extract_yes_no(response)
                return {"score": score_str, "reasoning": reasoning}
        
        except json.JSONDecodeError:
            # JSON解析失败，尝试提取JSON片段
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', response_stripped, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    if 'score' in json_data:
                        score = json_data['score']
                        reasoning = json_data.get('reasoning', '')
                        
                        # 检查是否为列表格式
                        if isinstance(score, list) and len(score) >= 2:
                            try:
                                naturalness = float(score[0])
                                artifacts = float(score[1])
                                naturalness = max(0.0, min(10.0, naturalness))
                                artifacts = max(0.0, min(10.0, artifacts))
                                return {"score": [naturalness, artifacts], "reasoning": reasoning}
                            except (ValueError, IndexError, TypeError):
                                pass
                        
                        # 否则使用yes/no格式
                        score_str = self.extract_yes_no(response)
                        return {"score": score_str, "reasoning": reasoning}
                except:
                    pass
        except Exception:
            pass
        
        # ===== 方法2: 向后兼容 - 纯文本格式 =====
        score_str = self.extract_yes_no(response)
        reasoning = self.extract_reasoning(response)
        return {"score": score_str, "reasoning": reasoning}
    
    def extract_reasoning(self, response: str) -> str:
        """
        从响应中提取reasoning字段
        
        Returns:
            reasoning字符串，如果未找到则返回空字符串
        """
        response_stripped = response.strip()
        
        # 尝试解析JSON格式
        try:
            json_data = json.loads(response_stripped)
            if 'reasoning' in json_data:
                reasoning = json_data['reasoning']
                if isinstance(reasoning, str):
                    return reasoning
                elif isinstance(reasoning, list):
                    # 如果reasoning是列表，拼接成字符串
                    return " ".join(str(r) for r in reasoning)
                else:
                    return str(reasoning)
        except json.JSONDecodeError:
            # JSON解析失败，尝试提取JSON片段
            json_match = re.search(r'\{[^}]*"reasoning"[^}]*\}', response_stripped, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    if 'reasoning' in json_data:
                        reasoning = json_data['reasoning']
                        if isinstance(reasoning, str):
                            return reasoning
                        else:
                            return str(reasoning)
                except:
                    pass
        except Exception:
            pass
        
        # 如果JSON解析失败，返回空字符串
        return ""
    
    def extract_yes_no(self, response: str) -> str:
        """
        从响应中提取yes/no二元结果
        
        支持两种格式：
        1. JSON格式: {"reasoning": "...", "score": 0或1}
        2. 纯文本格式: "yes" 或 "no" （向后兼容）
        
        Returns:
            "yes" 或 "no"
        """
        response_stripped = response.strip()
        
        # ===== 方法1: 尝试解析JSON格式 =====
        try:
            # 尝试直接解析JSON
            json_data = json.loads(response_stripped)
            
            # 提取score字段
            if 'score' in json_data:
                score = json_data['score']
                
                # 处理不同的score格式
                if isinstance(score, int):
                    # score为整数：0或1
                    if score == 1:
                        return "yes"
                    elif score == 0:
                        return "no"
                    else:
                        print(f"[Warning] Invalid score value: {score}, expected 0 or 1", 
                              file=sys.stderr, flush=True)
                        return "no"
                
                elif isinstance(score, str):
                    # score为字符串："0"或"1"
                    score_lower = score.strip().lower()
                    if score_lower == "1" or score_lower == "yes":
                        return "yes"
                    elif score_lower == "0" or score_lower == "no":
                        return "no"
                    else:
                        print(f"[Warning] Invalid score string: {score}", 
                              file=sys.stderr, flush=True)
                        return "no"
                
                elif isinstance(score, list) and len(score) > 0:
                    # score为列表：取第一个元素
                    score_val = score[0]
                    if score_val == 1 or str(score_val).lower() == "yes":
                        return "yes"
                    elif score_val == 0 or str(score_val).lower() == "no":
                        return "no"
                    else:
                        print(f"[Warning] Invalid score in list: {score_val}", 
                              file=sys.stderr, flush=True)
                        return "no"
                
                else:
                    print(f"[Warning] Unsupported score format: {type(score)}", 
                          file=sys.stderr, flush=True)
                    return "no"
            else:
                print(f"[Warning] JSON missing 'score' field", file=sys.stderr, flush=True)
                return "no"
        
        except json.JSONDecodeError:
            # JSON解析失败，尝试提取JSON片段
            # 有时模型输出可能包含额外文本，尝试提取JSON部分
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', response_stripped, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    if 'score' in json_data:
                        score = json_data['score']
                        if score == 1 or str(score).lower() == "yes":
                            return "yes"
                        elif score == 0 or str(score).lower() == "no":
                            return "no"
                except:
                    pass  # 继续尝试其他方法
        
        except Exception as e:
            print(f"[Warning] JSON parsing error: {e}", file=sys.stderr, flush=True)
        
        # ===== 方法2: 向后兼容 - 纯文本格式匹配 =====
        response_lower = response_stripped.lower()
        
        # 优先匹配独立的yes/no单词（避免匹配到其他单词中的字母）
        patterns_yes = [
            r'\byes\b',  # 独立的yes单词
            r'^yes\s*$',  # 整个响应就是yes
            r'answer[:\s]+yes',  # Answer: yes
            r'result[:\s]+yes',  # Result: yes
            r'output[:\s]+yes',  # Output: yes
            r'response[:\s]+yes',  # Response: yes
        ]
        
        patterns_no = [
            r'\bno\b',  # 独立的no单词
            r'^no\s*$',  # 整个响应就是no
            r'answer[:\s]+no',  # Answer: no
            r'result[:\s]+no',  # Result: no
            r'output[:\s]+no',  # Output: no
            r'response[:\s]+no',  # Response: no
        ]
        
        # 先检查yes（优先级更高，因为如果同时出现yes和no，倾向于yes）
        for pattern in patterns_yes:
            if re.search(pattern, response_lower):
                return "yes"
        
        # 再检查no
        for pattern in patterns_no:
            if re.search(pattern, response_lower):
                return "no"
        
        # ===== 方法3: 最后尝试提取0/1数字 =====
        # 提取独立的0或1数字
        score_match = re.search(r'\b([01])\b', response_stripped)
        if score_match:
            score_val = int(score_match.group(1))
            if score_val == 1:
                return "yes"
            elif score_val == 0:
                return "no"
        
        # 如果所有方法都失败，记录原始响应用于调试，返回默认值"no"
        print(f"[Warning] Could not extract yes/no from: '{response[:100]}'", file=sys.stderr, flush=True)
        return "no"  # 默认返回"no"（更保守）
    
    def score_single(self, edited_image_b64: str, system_prompt: str, 
                    user_prompt: str, rationale: str = None, 
                    original_image_b64: str = None, max_new_tokens: int = 128) -> dict:
        """
        评分单张图像
        
        Args:
            edited_image_b64: Base64编码的编辑后图像
            system_prompt: 系统提示
            user_prompt: 用户提示
            rationale: 编辑任务难点描述（可选）
            original_image_b64: Base64编码的原始图像（可选，用于对比评估）
            max_new_tokens: 最大生成token数
            
        Returns:
            评分结果字典 {"score": "yes"/"no", "reasoning": "..."}
        """
        # 如果提供了rationale，将其添加到user_prompt中
        if rationale:
            user_prompt = f"Task Challenge: {rationale}\n\n{user_prompt}"
        
        # 解码图像
        edited_image = self.decode_base64_image(edited_image_b64)
        
        # 构建messages - 明确标识原图和编辑后图像
        user_content = []
        
        if original_image_b64 is not None:
            # 如果提供了原图，先显示原图，再显示编辑后图像
            original_image = self.decode_base64_image(original_image_b64)
            user_content.extend([
                {"type": "text", "text": "Original image (before editing):"},
                {"type": "image", "image": original_image},
                {"type": "text", "text": "Edited image (after editing):"},
                {"type": "image", "image": edited_image},
                {"type": "text", "text": user_prompt}
            ])
        else:
            # 如果没有原图，只显示编辑后图像
            user_content.extend([
                {"type": "image", "image": edited_image},
                {"type": "text", "text": user_prompt}
            ])
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": user_content
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
        
        # 生成
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        
        # 提取score和reasoning
        result = self.extract_score_and_reasoning(output_text)
        return result
    
    def score_batch(self, tasks: List[Dict], batch_size: int = 4, 
                   max_new_tokens: int = 128, use_batch_inference: bool = True) -> List[Dict]:
        """
        批量评分
        
        Args:
            tasks: 任务列表，每个任务包含 edited_image_b64, system_prompt, user_prompt, 
                   rationale（可选）, original_image_b64（可选）
            batch_size: 批处理大小
            max_new_tokens: 最大生成token数
            use_batch_inference: 是否使用batch inference
            
        Returns:
            评分结果列表（每个元素为字典 {"score": "yes"/"no", "reasoning": "..."}）
        """
        n = len(tasks)
        
        if not use_batch_inference or batch_size == 1:
            # 串行处理
            print(f"[Qwen3VL-Standalone] Serial scoring {n} images", file=sys.stderr, flush=True)
            print(f"\n{'='*70}", file=sys.stderr, flush=True)
            print(f"[Qwen3-VL Scoring] Starting serial scoring for {n} images", file=sys.stderr, flush=True)
            print(f"{'='*70}\n", file=sys.stderr, flush=True)
            
            results = []
            for i, task in enumerate(tasks):
                # 支持新的字段名 edited_image_b64，也兼容旧的 image_b64
                edited_image_b64 = task.get('edited_image_b64', task.get('image_b64'))
                original_image_b64 = task.get('original_image_b64', None)
                
                result = self.score_single(
                    edited_image_b64,
                    task['system_prompt'],
                    task['user_prompt'],
                    task.get('rationale', None),
                    original_image_b64,  # 传入原图
                    max_new_tokens
                )
                results.append(result)
                
                # 打印每个样本的详细信息
                score_str = result.get('score', 'N/A')
                reasoning = result.get('reasoning', '')
                reasoning_preview = reasoning[:80] if reasoning else "N/A"
                
                # 处理列表类型的score（PQ/SC指标）
                if isinstance(score_str, list):
                    score_display = f"{score_str}"
                else:
                    score_display = score_str
                
                print(f"  [Sample {i+1:3d}/{n}] Score: {score_display} | Reasoning: {reasoning_preview}...", 
                      file=sys.stderr, flush=True)
            
            # 打印评分总结
            if results:
                # 统计yes/no（对于字符串类型的score）
                yes_count = sum(1 for r in results if isinstance(r.get('score'), str) and r.get('score', '').lower() == "yes")
                no_count = sum(1 for r in results if isinstance(r.get('score'), str) and r.get('score', '').lower() == "no")
                total_str_scores = yes_count + no_count
                
                print(f"\n{'='*70}", file=sys.stderr, flush=True)
                print(f"[Qwen3-VL Scoring] Serial scoring completed!", file=sys.stderr, flush=True)
                print(f"  Total images: {len(results)}", file=sys.stderr, flush=True)
                if total_str_scores > 0:
                    yes_rate = (yes_count / total_str_scores * 100) if total_str_scores > 0 else 0.0
                    print(f"  Yes count: {yes_count} ({yes_rate:.2f}%)", file=sys.stderr, flush=True)
                    print(f"  No count: {no_count} ({100-yes_rate:.2f}%)", file=sys.stderr, flush=True)
                print(f"{'='*70}\n", file=sys.stderr, flush=True)
            
            return results
        
        # Batch inference
        print(f"[Qwen3VL-Standalone] Batch scoring {n} images with batch_size={batch_size}", 
              file=sys.stderr, flush=True)
        
        # 设置padding_side
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = 'left'
        
        all_scores = []
        
        # 打印评分开始信息
        print(f"\n{'='*70}", file=sys.stderr, flush=True)
        print(f"[Qwen3-VL Scoring] Starting batch scoring for {n} images", file=sys.stderr, flush=True)
        print(f"  Batch size: {batch_size}", file=sys.stderr, flush=True)
        print(f"  Total batches: {(n + batch_size - 1) // batch_size}", file=sys.stderr, flush=True)
        print(f"{'='*70}\n", file=sys.stderr, flush=True)
        
        try:
            # 分批处理
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                batch_tasks = tasks[batch_start:batch_end]
                
                # 解码图像（支持新的字段名，也兼容旧的）
                edited_images = []
                original_images = []
                for task in batch_tasks:
                    edited_image_b64 = task.get('edited_image_b64', task.get('image_b64'))
                    edited_images.append(self.decode_base64_image(edited_image_b64))
                    
                    original_image_b64 = task.get('original_image_b64', None)
                    if original_image_b64:
                        original_images.append(self.decode_base64_image(original_image_b64))
                    else:
                        original_images.append(None)
                
                # 构建batch messages
                batch_messages = []
                for task, edited_image, original_image in zip(batch_tasks, edited_images, original_images):
                    # 如果提供了rationale，将其添加到user_prompt中
                    user_prompt = task['user_prompt']
                    rationale = task.get('rationale', None)
                    if rationale:
                        user_prompt = f"Task Challenge: {rationale}\n\n{user_prompt}"
                    
                    # 构建user content，明确标识原图和编辑后图像
                    user_content = []
                    
                    if original_image is not None:
                        # 如果提供了原图，先显示原图，再显示编辑后图像
                        user_content.extend([
                            {"type": "text", "text": "Original image (before editing):"},
                            {"type": "image", "image": original_image},
                            {"type": "text", "text": "Edited image (after editing):"},
                            {"type": "image", "image": edited_image},
                            {"type": "text", "text": user_prompt}
                        ])
                    else:
                        # 如果没有原图，只显示编辑后图像
                        user_content.extend([
                            {"type": "image", "image": edited_image},
                            {"type": "text", "text": user_prompt}
                        ])
                    
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": task['system_prompt']}]
                        },
                        {
                            "role": "user",
                            "content": user_content
                        }
                    ]
                    batch_messages.append(messages)
                
                # 准备输入
                inputs = self.processor.apply_chat_template(
                    batch_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True
                )
                inputs = inputs.to(self.model.device)
                
                # 生成
                with torch.inference_mode():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] 
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_texts = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                
                # 提取score和reasoning结果并打印详细信息
                batch_results = []
                for i, (text, task) in enumerate(zip(output_texts, batch_tasks)):
                    result_dict = self.extract_score_and_reasoning(text)
                    batch_results.append(result_dict)
                    
                    # 打印每个样本的详细信息
                    global_idx = batch_start + i
                    reasoning_preview = result_dict['reasoning'][:50] if result_dict['reasoning'] else "N/A"
                    print(f"  [Sample {global_idx:3d}] Score: {result_dict['score']} | Reasoning: {reasoning_preview}...", 
                          file=sys.stderr, flush=True)
                
                all_scores.extend(batch_results)
                
                # 打印批次统计
                yes_count = sum(1 for r in batch_results if r['score'] == "yes")
                no_count = len(batch_results) - yes_count
                print(f"[Batch {batch_start//batch_size + 1}] Images {batch_start}-{batch_end-1} done, "
                      f"yes={yes_count}, no={no_count}", 
                      file=sys.stderr, flush=True)
        
        finally:
            # 恢复padding_side
            self.processor.tokenizer.padding_side = original_padding_side
        
        # 打印评分总结
        if all_scores:
            yes_count = sum(1 for r in all_scores if r['score'] == "yes")
            no_count = len(all_scores) - yes_count
            yes_rate = (yes_count / len(all_scores) * 100) if all_scores else 0.0
            print(f"\n{'='*70}", file=sys.stderr, flush=True)
            print(f"[Qwen3-VL Scoring] Completed!", file=sys.stderr, flush=True)
            print(f"  Total images: {len(all_scores)}", file=sys.stderr, flush=True)
            print(f"  Yes count: {yes_count} ({yes_rate:.2f}%)", file=sys.stderr, flush=True)
            print(f"  No count: {no_count} ({100-yes_rate:.2f}%)", file=sys.stderr, flush=True)
            print(f"{'='*70}\n", file=sys.stderr, flush=True)
        
        return all_scores


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Standalone Scorer")
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--model-name', default='Qwen/Qwen3-VL-30B-Instruct', 
                       help='Model name or path')
    parser.add_argument('--device', default='auto', help='Device: auto, cuda, cpu')
    parser.add_argument('--dtype', default='bfloat16', 
                       help='Data type: bfloat16, float16, float32, auto')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--max-new-tokens', type=int, default=128, 
                       help='Max new tokens')
    parser.add_argument('--use-batch-inference', action='store_true', default=True,
                       help='Use batch inference')
    
    args = parser.parse_args()
    
    try:
        # 读取输入
        print(f"[Qwen3VL-Standalone] Reading input from: {args.input}", file=sys.stderr, flush=True)
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # 初始化模型
        scorer = Qwen3VLStandaloneScorer(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype
        )
        
        # 评分
        tasks = input_data.get('tasks', [])
        print(f"[Qwen3VL-Standalone] Processing {len(tasks)} tasks...", file=sys.stderr, flush=True)
        
        scores = scorer.score_batch(
            tasks=tasks,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            use_batch_inference=args.use_batch_inference
        )
        
        # 写入输出（scores现在是字典列表）
        output_data = {
            'scores': scores,  # 包含 {"score": "yes/no", "reasoning": "..."} 的列表
            'status': 'success',
            'num_tasks': len(tasks)
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"[Qwen3VL-Standalone] Results written to: {args.output}", file=sys.stderr, flush=True)
        yes_count = sum(1 for s in scores if s['score'] == "yes")
        yes_rate = (yes_count / len(scores) * 100) if scores else 0.0
        print(f"[Qwen3VL-Standalone] Yes rate: {yes_rate:.2f}% ({yes_count}/{len(scores)})", 
              file=sys.stderr, flush=True)
        
        sys.exit(0)
    
    except Exception as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        # 写入错误信息
        error_data = {
            'status': 'error',
            'error': str(e),
            'scores': []  # 保持字段名不变以兼容
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        
        sys.exit(1)


if __name__ == '__main__':
    main()

