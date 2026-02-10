#!/usr/bin/env python3
"""
Qwen3-VL vLLM Standalone评分脚本
用于在独立虚拟环境（yx_vllm）中运行Qwen3-VL模型（基于vLLM引擎）

复用 single_scoring_part_vllm 的核心组件：
- VLLMModel 初始化逻辑
- OutputParser 解析逻辑
- prepare_inputs_for_vllm 输入准备函数

使用方法：
    conda run -n yx_vllm python qwen3_vl_vllm_standalone.py --input input.json --output output.json
"""

import argparse
import json
import base64
import sys
import os
import re
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Optional, Union

# 设置环境变量（vLLM要求）
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# 在新环境中导入vLLM和相关依赖
try:
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    from PIL import Image
except ImportError as e:
    print(f"Error: Failed to import required packages: {e}", file=sys.stderr, flush=True)
    print("Please ensure yx_vllm environment has: vllm, qwen_vl_utils, transformers, torch, pillow", file=sys.stderr, flush=True)
    sys.exit(1)


# ================================
# OutputParser（复用自 single_scoring_part_vllm/output_parser.py）
# ================================

class OutputParser:
    """输出解析器"""
    
    def __init__(self):
        pass
    
    def parse_output(self, generated_text: str) -> Dict[str, Any]:
        """
        解析模型生成的文本，提取JSON格式的输出
        
        Args:
            generated_text: 模型生成的文本
            
        Returns:
            解析后的字典，包含score和reasoning字段
        """
        try:
            json_str = self._extract_json(generated_text)
            
            if json_str:
                parsed = json.loads(json_str)
                
                if "score" not in parsed:
                    return {
                        "score": None,
                        "reasoning": parsed.get("reasoning", parsed.get("reason", "")),
                        "error": "Missing 'score' field"
                    }
                
                # 支持"reasoning"和"reason"两种字段名
                reasoning = parsed.get("reasoning", parsed.get("reason", ""))
                
                return {
                    "score": parsed["score"],
                    "reasoning": reasoning,
                    "raw_output": generated_text
                }
            else:
                return {
                    "score": None,
                    "reasoning": "",
                    "error": "No JSON found in output",
                    "raw_output": generated_text
                }
                
        except json.JSONDecodeError as e:
            return {
                "score": None,
                "reasoning": "",
                "error": f"JSON decode error: {str(e)}",
                "raw_output": generated_text
            }
        except Exception as e:
            return {
                "score": None,
                "reasoning": "",
                "error": f"Parse error: {str(e)}",
                "raw_output": generated_text
            }
    
    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取JSON部分"""
        # 方法1: 尝试找到 {...} 格式的JSON
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            json_str = max(matches, key=len)
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
        """标准化score值"""
        if score is None:
            return None
        
        if isinstance(score, str):
            score_lower = score.lower().strip()
            if score_lower in ['yes', 'no']:
                return score_lower
            try:
                return float(score)
            except:
                return score
        
        if isinstance(score, list):
            try:
                return [float(x) for x in score]
            except:
                return score
        
        if isinstance(score, (int, float)):
            return float(score)
        
        return score


# ================================
# vLLM输入准备函数（复用自 single_scoring_part_vllm/vllm_model.py）
# ================================

def prepare_inputs_for_vllm(messages: List[Dict], processor: AutoProcessor) -> Dict:
    """
    准备vLLM的输入格式
    
    Args:
        messages: 消息列表
        processor: AutoProcessor实例
        
    Returns:
        包含prompt、multi_modal_data、mm_processor_kwargs的字典
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


# ================================
# vLLM评分器（整合所有功能）
# ================================

class Qwen3VLvLLMStandaloneScorer:
    """独立的Qwen3-VL vLLM评分器"""
    
    def __init__(self, 
                 model_name: str,
                 tensor_parallel_size: int,
                 batch_size: int,
                 max_new_tokens: int,
                 temperature: float = 0,
                 dtype: str = "bfloat16"):
        """
        初始化vLLM模型
        
        Args:
            model_name: 模型名称或路径
            tensor_parallel_size: tensor并行大小
            batch_size: 批处理大小
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            dtype: 数据类型
        """
        print(f"[Qwen3VL-vLLM-Standalone] Loading model: {model_name}", file=sys.stderr, flush=True)
        print(f"[Qwen3VL-vLLM-Standalone] Tensor parallel size: {tensor_parallel_size}", file=sys.stderr, flush=True)
        print(f"[Qwen3VL-vLLM-Standalone] Batch size: {batch_size}", file=sys.stderr, flush=True)
        
        # 加载processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # 初始化LLM（vLLM引擎）
        self.llm = LLM(
            model=model_name,
            mm_encoder_tp_mode="data",  # 多模态编码器并行模式
            enable_expert_parallel=True,
            tensor_parallel_size=tensor_parallel_size,
            seed=0
        )
        
        # 初始化采样参数
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_k=-1,
            stop_token_ids=[],
        )
        
        self.batch_size = batch_size
        self.output_parser = OutputParser()
        
        print(f"[Qwen3VL-vLLM-Standalone] Model loaded successfully", file=sys.stderr, flush=True)
    
    def decode_base64_image(self, base64_str: str) -> str:
        """
        处理base64图像字符串，确保是data URI格式
        
        Args:
            base64_str: base64字符串或data URI
            
        Returns:
            data URI格式字符串
        """
        if base64_str.startswith("data:image/"):
            # 已经是data URI格式
            return base64_str
        else:
            # 纯base64字符串，转换为data URI
            return f"data:image/png;base64,{base64_str}"
    
    def score_batch(self, tasks: List[Dict]) -> List[Dict[str, Any]]:
        """
        批量评分
        
        Args:
            tasks: 任务列表，每个任务包含：
                - edited_image_b64: 编辑后图像（base64或data URI）
                - original_image_b64: 原始图像（可选）
                - system_prompt: 系统提示
                - user_prompt: 用户提示
                - rationale: 任务难点（可选）
                
        Returns:
            评分结果列表
        """
        print(f"[Qwen3VL-vLLM-Standalone] Processing {len(tasks)} tasks...", file=sys.stderr, flush=True)
        
        all_results = []
        
        # 分批处理
        for i in range(0, len(tasks), self.batch_size):
            batch_tasks = tasks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tasks) + self.batch_size - 1) // self.batch_size
            
            print(f"[Qwen3VL-vLLM-Standalone] Processing batch {batch_num}/{total_batches} ({len(batch_tasks)} tasks)", 
                  file=sys.stderr, flush=True)
            
            # 准备批量messages
            batch_messages = []
            for task in batch_tasks:
                # 准备图像列表
                images = []
                
                # 添加原图（如果有）
                if 'original_image_b64' in task and task['original_image_b64']:
                    original_img = self.decode_base64_image(task['original_image_b64'])
                    images.append(original_img)
                
                # 添加编辑后的图像
                edited_img = self.decode_base64_image(task['edited_image_b64'])
                images.append(edited_img)
                
                # 构建user prompt（注入rationale）
                user_prompt = task['user_prompt']
                if task.get('rationale') and task['rationale']:
                    # 检查user_prompt中是否已经有rationale占位符
                    if '{rationale}' in user_prompt:
                        user_prompt = user_prompt.replace('{rationale}', f"\n\nTask Challenge: {task['rationale']}")
                    else:
                        # 直接追加到末尾
                        user_prompt = user_prompt + f"\n\nTask Challenge: {task['rationale']}"
                
                # 构建messages
                messages = [
                    {"role": "system", "content": task['system_prompt']},
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": img} for img in images],
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
                
                batch_messages.append(messages)
            
            # 准备vLLM输入
            vllm_inputs = []
            for messages in batch_messages:
                input_dict = prepare_inputs_for_vllm(messages, self.processor)
                vllm_inputs.append(input_dict)
            
            # vLLM批量生成
            try:
                outputs = self.llm.generate(vllm_inputs, self.sampling_params)
                
                # 解析输出
                for output in outputs:
                    generated_text = output.outputs[0].text
                    parsed = self.output_parser.parse_output(generated_text)
                    
                    # 标准化score
                    if parsed["score"] is not None:
                        parsed["score"] = self.output_parser.normalize_score(parsed["score"])
                    
                    all_results.append(parsed)
                
                print(f"[Qwen3VL-vLLM-Standalone] Batch {batch_num}/{total_batches} completed", 
                      file=sys.stderr, flush=True)
                
            except Exception as e:
                print(f"[Qwen3VL-vLLM-Standalone] Error in batch {batch_num}: {e}", 
                      file=sys.stderr, flush=True)
                # 为失败的批次添加错误结果
                for _ in batch_tasks:
                    all_results.append({
                        "score": None,
                        "reasoning": "",
                        "error": str(e)
                    })
        
        print(f"[Qwen3VL-vLLM-Standalone] All tasks completed: {len(all_results)} results", 
              file=sys.stderr, flush=True)
        
        return all_results


# ================================
# 主函数
# ================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL vLLM Standalone Scorer")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--model-name", type=str, required=True, help="Model name or path")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallel size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    
    args = parser.parse_args()
    
    # 自动检测tensor_parallel_size
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = torch.cuda.device_count()
        print(f"[Qwen3VL-vLLM-Standalone] Auto-detected tensor_parallel_size: {args.tensor_parallel_size}", 
              file=sys.stderr, flush=True)
    
    try:
        # 读取输入数据
        print(f"[Qwen3VL-vLLM-Standalone] Loading input from: {args.input}", file=sys.stderr, flush=True)
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        tasks = input_data.get('tasks', [])
        print(f"[Qwen3VL-vLLM-Standalone] Loaded {len(tasks)} tasks", file=sys.stderr, flush=True)
        
        # 初始化评分器
        scorer = Qwen3VLvLLMStandaloneScorer(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            dtype=args.dtype
        )
        
        # 批量评分
        results = scorer.score_batch(tasks)
        
        # 写入输出
        output_data = {
            'status': 'success',
            'scores': results
        }
        
        print(f"[Qwen3VL-vLLM-Standalone] Writing output to: {args.output}", file=sys.stderr, flush=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"[Qwen3VL-vLLM-Standalone] Success: {len(results)} results written", file=sys.stderr, flush=True)
        
    except Exception as e:
        print(f"[Qwen3VL-vLLM-Standalone] Fatal error: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        # 写入错误输出
        output_data = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()
