#!/usr/bin/env python
"""
Step1X-Edit Subprocess Worker
在特定虚拟环境中运行 Step1X-Edit 模型推理

用法:
    python step1x_edit_subprocess_worker.py --input input.json --output output.json --model-name ... --device cuda:0
"""

import argparse
import json
import base64
import sys
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


def main():
    parser = argparse.ArgumentParser(description='Step1X-Edit Subprocess Worker')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--model-name', required=True, help='Model name or path')
    parser.add_argument('--device', default='cuda:0', help='Device (e.g., cuda:0)')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--true-cfg-scale', type=float, default=6.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--enable-thinking-mode', action='store_true')
    parser.add_argument('--enable-reflection-mode', action='store_true')
    parser.add_argument('--disable-progress-bar', action='store_true')
    
    args = parser.parse_args()
    
    # 解析设备
    device = args.device
    gpu_id = int(device.split(':')[1]) if ':' in device else 0
    
    print(f"[Worker] Starting on {device}", file=sys.stderr)
    print(f"[Worker] Model: {args.model_name}", file=sys.stderr)
    
    try:
        # 读取输入数据
        with open(args.input, 'r') as f:
            input_data = json.load(f)
        
        tasks = input_data['tasks']
        print(f"[Worker] Received {len(tasks)} tasks", file=sys.stderr)
        
        # 设置设备
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        
        # 解析dtype
        if args.dtype == 'bfloat16':
            torch_dtype = torch.bfloat16
        elif args.dtype == 'float16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # 加载模型
        print(f"[Worker] Loading Step1X-Edit pipeline...", file=sys.stderr)
        from diffusers import Step1XEditPipelineV1P2
        
        pipeline = Step1XEditPipelineV1P2.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
        )
        pipeline.to(device)
        
        if args.disable_progress_bar:
            pipeline.set_progress_bar_config(disable=True)
        
        print(f"[Worker] Model loaded successfully", file=sys.stderr)
        
        # 处理任务
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
                
                # 准备生成器
                generator = torch.Generator(device=device).manual_seed(current_seed)
                
                # 执行推理
                with torch.inference_mode():
                    pipe_output = pipeline(
                        image=image,
                        prompt=instruction,
                        num_inference_steps=args.num_inference_steps,
                        true_cfg_scale=args.true_cfg_scale,
                        generator=generator,
                        enable_thinking_mode=args.enable_thinking_mode,
                        enable_reflection_mode=args.enable_reflection_mode,
                    )
                    edited_image = pipe_output.images[0]
                
                # 编码结果
                result_b64 = encode_image(edited_image)
                results.append({
                    'task_id': task_id,
                    'success': True,
                    'image_b64': result_b64,
                    'error': None
                })
                
                print(f"[Worker] Task {task_id} completed ✓", file=sys.stderr)
                
            except Exception as e:
                print(f"[Worker] Task {task_id} failed: {e}", file=sys.stderr)
                results.append({
                    'task_id': task_id,
                    'success': False,
                    'image_b64': None,
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

