#!/usr/bin/env python
"""
DreamOmni2 Subprocess Worker
在特定虚拟环境中运行 DreamOmni2 模型推理

用法:
    python dreamomni2_subprocess_worker.py --input input.json --output output.json ...

DreamOmni2 推理流程：
1. 加载 VLM (Qwen2.5-VL) 和 Diffusion (FLUX.1-Kontext + LoRA)
2. VLM 处理图像和指令，生成 prompt
3. Diffusion 根据 prompt 生成编辑后的图像
"""

import argparse
import json
import base64
import sys
import os
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image


def decode_image(b64_str: str) -> Image.Image:
    """将base64字符串解码为PIL Image"""
    image_data = base64.b64decode(b64_str)
    return Image.open(BytesIO(image_data))


def encode_image(image: Image.Image) -> str:
    """将PIL Image编码为base64字符串"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def extract_gen_content(text: str) -> str:
    """提取生成内容（去除特殊标记）"""
    # 官方实现：去除前6个和后7个字符
    if len(text) > 13:
        return text[6:-7]
    return text


def main():
    parser = argparse.ArgumentParser(description='DreamOmni2 Subprocess Worker')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--vlm-path', required=True, help='VLM model path')
    parser.add_argument('--base-model-path', required=True, help='Base diffusion model path')
    parser.add_argument('--edit-lora-path', required=True, help='Edit LoRA weights path')
    parser.add_argument('--dreamomni2-repo', required=True, help='DreamOmni2 repository path')
    parser.add_argument('--device', default='cuda:0', help='Device (e.g., cuda:0)')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--num-inference-steps', type=int, default=30)
    parser.add_argument('--guidance-scale', type=float, default=3.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--disable-progress-bar', action='store_true')
    
    args = parser.parse_args()
    
    # 解析设备
    device = args.device
    gpu_id = int(device.split(':')[1]) if ':' in device else 0
    
    print(f"[Worker] Starting DreamOmni2 on {device}", file=sys.stderr)
    print(f"[Worker] VLM: {args.vlm_path}", file=sys.stderr)
    print(f"[Worker] Base Model: {args.base_model_path}", file=sys.stderr)
    print(f"[Worker] Edit LoRA: {args.edit_lora_path}", file=sys.stderr)
    
    try:
        # 读取输入数据
        with open(args.input, 'r') as f:
            input_data = json.load(f)
        
        tasks = input_data['tasks']
        print(f"[Worker] Received {len(tasks)} tasks", file=sys.stderr)
        
        # 设置设备
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        
        # 添加 DreamOmni2 仓库到 Python 路径
        dreamomni2_repo = Path(args.dreamomni2_repo)
        if str(dreamomni2_repo) not in sys.path:
            sys.path.insert(0, str(dreamomni2_repo))
        
        print(f"[Worker] Added DreamOmni2 repo to path: {dreamomni2_repo}", file=sys.stderr)
        
        # 解析dtype
        if args.dtype == 'bfloat16':
            torch_dtype = torch.bfloat16
        elif args.dtype == 'float16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # ===== 加载 DreamOmni2 Pipeline =====
        print(f"[Worker] Loading DreamOmni2 Pipeline...", file=sys.stderr)
        from dreamomni2.pipeline_dreamomni2 import DreamOmni2Pipeline
        
        pipe = DreamOmni2Pipeline.from_pretrained(
            args.base_model_path, 
            torch_dtype=torch_dtype
        )
        pipe.to(device)
        
        # 加载 Edit LoRA
        print(f"[Worker] Loading Edit LoRA...", file=sys.stderr)
        pipe.load_lora_weights(
            args.edit_lora_path,
            adapter_name="edit"
        )
        pipe.set_adapters(["edit"], adapter_weights=[1])
        
        print(f"[Worker] DreamOmni2 Pipeline loaded successfully", file=sys.stderr)
        
        # ===== 加载 VLM =====
        print(f"[Worker] Loading VLM (Qwen2.5-VL)...", file=sys.stderr)
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from utils.vprocess import process_vision_info, resizeinput
        
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.vlm_path, 
            torch_dtype=torch_dtype, 
            device_map=device
        )
        processor = AutoProcessor.from_pretrained(args.vlm_path)
        
        print(f"[Worker] VLM loaded successfully", file=sys.stderr)
        
        # 定义 VLM 推理函数
        def infer_vlm(image: Image.Image, instruction: str) -> str:
            """使用 VLM 生成 prompt"""
            # 保存临时图像（VLM 需要路径或 PIL Image）
            # 构建消息格式
            prefix = " It is editing task."
            tp = [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction + prefix}
            ]
            messages = [{"role": "user", "content": tp}]
            
            # 准备推理
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            
            # 推理
            with torch.inference_mode():
                generated_ids = vlm_model.generate(
                    **inputs, 
                    do_sample=False, 
                    max_new_tokens=4096
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            return output_text[0]
        
        # 定义 Diffusion 推理函数
        def infer_diffusion(source_img: Image.Image, prompt: str) -> Image.Image:
            """使用 Diffusion 生成编辑后的图像"""
            # Resize 输入图像
            resized_img = resizeinput(source_img)
            
            with torch.inference_mode():
                image = pipe(
                    images=[resized_img],
                    height=resized_img.height,
                    width=resized_img.width,
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                ).images[0]
            return image
        
        # ===== 处理任务 =====
        results = []
        for i, task in enumerate(tasks):
            task_id = task['task_id']
            image_b64 = task['image_b64']
            instruction = task['instruction']
            current_seed = task.get('seed', args.seed)
            
            print(f"[Worker] Processing task {i+1}/{len(tasks)} (id={task_id})", file=sys.stderr)
            
            try:
                # 解码图像
                image = decode_image(image_b64)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 设置随机种子
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)
                
                # Step 1: VLM 生成 prompt
                print(f"[Worker] Task {task_id}: Running VLM inference...", file=sys.stderr)
                vlm_output = infer_vlm(image, instruction)
                prompt = extract_gen_content(vlm_output)
                print(f"[Worker] Task {task_id}: VLM prompt generated (len={len(prompt)})", file=sys.stderr)
                
                # Step 2: Diffusion 生成图像
                print(f"[Worker] Task {task_id}: Running Diffusion inference...", file=sys.stderr)
                edited_image = infer_diffusion(image, prompt)
                
                # 编码结果
                result_b64 = encode_image(edited_image)
                results.append({
                    'task_id': task_id,
                    'success': True,
                    'image_b64': result_b64,
                    'vlm_prompt': prompt,  # 保存 VLM 生成的 prompt
                    'error': None
                })
                
                print(f"[Worker] Task {task_id} completed ✓", file=sys.stderr)
                
            except Exception as e:
                print(f"[Worker] Task {task_id} failed: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                results.append({
                    'task_id': task_id,
                    'success': False,
                    'image_b64': None,
                    'vlm_prompt': None,
                    'error': str(e)
                })
            
            # 清理缓存
            torch.cuda.empty_cache()
        
        # 写入输出
        output_data = {
            'status': 'success',
            'results': results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f)
        
        print(f"[Worker] All tasks completed", file=sys.stderr)
        
    except Exception as e:
        print(f"[Worker] Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        # 写入错误输出
        output_data = {
            'status': 'error',
            'error': str(e),
            'results': []
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f)
        
        sys.exit(1)


if __name__ == '__main__':
    main()

