#!/usr/bin/env python
"""
OmniGen2 Subprocess Worker
在特定虚拟环境中运行 OmniGen2 模型推理

用法:
    python omnigen2_subprocess_worker.py --input input.json --output output.json ...

OmniGen2 推理流程：
1. 加载 OmniGen2Pipeline
2. 使用 text_guidance_scale 和 image_guidance_scale 进行推理
3. 输出编辑后的图像
"""

import argparse
import json
import base64
import sys
import os
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageOps


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
    parser = argparse.ArgumentParser(description='OmniGen2 Subprocess Worker')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--model-path', required=True, help='OmniGen2 model path')
    parser.add_argument('--transformer-path', default=None, help='Transformer checkpoint path')
    parser.add_argument('--transformer-lora-path', default=None, help='Transformer LoRA path')
    parser.add_argument('--omnigen2-repo', required=True, help='OmniGen2 repository path')
    parser.add_argument('--device', default='cuda:0', help='Device (e.g., cuda:0)')
    parser.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--text-guidance-scale', type=float, default=5.0)
    parser.add_argument('--image-guidance-scale', type=float, default=2.0)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scheduler', default='euler', choices=['euler', 'dpmsolver++'])
    parser.add_argument('--negative-prompt', type=str, 
                        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar")
    parser.add_argument('--enable-teacache', action='store_true')
    parser.add_argument('--teacache-rel-l1-thresh', type=float, default=0.05)
    parser.add_argument('--enable-taylorseer', action='store_true')
    parser.add_argument('--disable-progress-bar', action='store_true')
    
    args = parser.parse_args()
    
    # 解析设备
    device = args.device
    gpu_id = int(device.split(':')[1]) if ':' in device else 0
    
    print(f"[Worker] Starting OmniGen2 on {device}", file=sys.stderr)
    print(f"[Worker] Model: {args.model_path}", file=sys.stderr)
    if args.transformer_path:
        print(f"[Worker] Transformer: {args.transformer_path}", file=sys.stderr)
    if args.transformer_lora_path:
        print(f"[Worker] Transformer LoRA: {args.transformer_lora_path}", file=sys.stderr)
    
    try:
        # 读取输入数据
        with open(args.input, 'r') as f:
            input_data = json.load(f)
        
        tasks = input_data['tasks']
        print(f"[Worker] Received {len(tasks)} tasks", file=sys.stderr)
        
        # 设置设备
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)  # 在CUDA_VISIBLE_DEVICES设置后，使用device 0
        torch.cuda.empty_cache()
        
        # 添加 OmniGen2 仓库到 Python 路径
        omnigen2_repo = Path(args.omnigen2_repo)
        if str(omnigen2_repo) not in sys.path:
            sys.path.insert(0, str(omnigen2_repo))
        
        print(f"[Worker] Added OmniGen2 repo to path: {omnigen2_repo}", file=sys.stderr)
        
        # 解析dtype
        if args.dtype == 'bf16':
            weight_dtype = torch.bfloat16
        elif args.dtype == 'fp16':
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
        
        # ===== 加载 OmniGen2 Pipeline =====
        print(f"[Worker] Loading OmniGen2 Pipeline...", file=sys.stderr)
        
        from accelerate import Accelerator
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
        
        # 初始化 Accelerator
        accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != 'fp32' else 'no')
        
        # 加载 Pipeline
        pipeline = OmniGen2Pipeline.from_pretrained(
            args.model_path,
            torch_dtype=weight_dtype,
            trust_remote_code=True,
        )
        
        # 加载 Transformer
        if args.transformer_path:
            print(f"[Worker] Loading transformer from {args.transformer_path}", file=sys.stderr)
            pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
                args.transformer_path,
                torch_dtype=weight_dtype,
            )
        else:
            pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
                args.model_path,
                subfolder="transformer",
                torch_dtype=weight_dtype,
            )
        
        # 加载 LoRA
        if args.transformer_lora_path:
            print(f"[Worker] Loading LoRA from {args.transformer_lora_path}", file=sys.stderr)
            pipeline.load_lora_weights(args.transformer_lora_path)
        
        # 配置加速选项
        if args.enable_teacache and args.enable_taylorseer:
            print("[Worker] WARNING: enable_teacache and enable_taylorseer are mutually exclusive. enable_teacache will be ignored.", file=sys.stderr)
        
        if args.enable_taylorseer:
            pipeline.enable_taylorseer = True
            print("[Worker] TaylorSeer enabled", file=sys.stderr)
        elif args.enable_teacache:
            pipeline.transformer.enable_teacache = True
            pipeline.transformer.teacache_rel_l1_thresh = args.teacache_rel_l1_thresh
            print(f"[Worker] TeaCache enabled (threshold={args.teacache_rel_l1_thresh})", file=sys.stderr)
        
        # 配置 scheduler
        if args.scheduler == "dpmsolver++":
            from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
            scheduler = DPMSolverMultistepScheduler(
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                solver_order=2,
                prediction_type="flow_prediction",
            )
            pipeline.scheduler = scheduler
            print("[Worker] Using DPMSolver++ scheduler", file=sys.stderr)
        
        # 移动到设备
        pipeline = pipeline.to(accelerator.device)
        
        print(f"[Worker] OmniGen2 Pipeline loaded successfully", file=sys.stderr)
        
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
                
                # EXIF处理
                image = ImageOps.exif_transpose(image)
                
                # 创建生成器
                generator = torch.Generator(device=accelerator.device).manual_seed(current_seed)
                
                # 运行推理
                print(f"[Worker] Task {task_id}: Running OmniGen2 inference...", file=sys.stderr)
                print(f"[Worker] Task {task_id}: Instruction: {instruction[:100]}...", file=sys.stderr)
                
                output = pipeline(
                    prompt=instruction,
                    input_images=[image],
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.num_inference_steps,
                    max_sequence_length=1024,
                    text_guidance_scale=args.text_guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    cfg_range=(0.0, 1.0),
                    negative_prompt=args.negative_prompt,
                    num_images_per_prompt=1,
                    generator=generator,
                    output_type="pil",
                )
                
                # 获取第一张输出图像
                edited_image = output.images[0]
                
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
                import traceback
                traceback.print_exc(file=sys.stderr)
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
















