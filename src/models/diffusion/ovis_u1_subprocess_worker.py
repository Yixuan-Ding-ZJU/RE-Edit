#!/usr/bin/env python
"""
Ovis-U1 Subprocess Worker
在特定虚拟环境中运行 Ovis-U1 模型推理

用法:
    python ovis_u1_subprocess_worker.py --input input.json --output output.json \
        --model-path AIDC-AI/Ovis-U1-3B --device cuda:0
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


def load_blank_image(width, height):
    """创建空白图像（用于无条件生成）"""
    pil_image = Image.new("RGB", (width, height), (255, 255, 255)).convert('RGB')
    return pil_image


def build_inputs(model, text_tokenizer, visual_tokenizer, prompt, pil_image, target_width, target_height, device):
    """构建模型输入"""
    if pil_image is not None:
        target_size = (int(target_width), int(target_height))
        pil_image, vae_pixel_values, cond_img_ids = model.visual_generator.process_image_aspectratio(pil_image, target_size)
        cond_img_ids[..., 0] = 1.0
        vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=device)
        width = pil_image.width
        height = pil_image.height
        resized_height, resized_width = visual_tokenizer.smart_resize(height, width, max_pixels=visual_tokenizer.image_processor.min_pixels)
        pil_image = pil_image.resize((resized_width, resized_height))
    else:
        vae_pixel_values = None
        cond_img_ids = None

    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt, 
        [pil_image], 
        generation_preface=None,
        return_labels=False,
        propagate_exception=False,
        multimodal_type='single_image',
        fix_sample_overall_length_navit=False
    )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=device)
    attention_mask = attention_mask.unsqueeze(0).to(device=device)
    if pixel_values is not None:
        pixel_values = torch.cat([
            pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16) if pixel_values is not None else None
        ], dim=0)
    if grid_thws is not None:
        grid_thws = torch.cat([
            grid_thws.to(device=visual_tokenizer.device) if grid_thws is not None else None
        ], dim=0)
    return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values


def pipe_img_edit(model, input_img, prompt, steps, txt_cfg, img_cfg, seed=42, device="cuda"):
    """执行图像编辑"""
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    
    width, height = input_img.size
    height, width = visual_tokenizer.smart_resize(height, width, factor=32)

    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        repetition_penalty=None,
        eos_token_id=text_tokenizer.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True,
        height=height,
        width=width,
        num_steps=steps,
        seed=seed,
        img_cfg=img_cfg,
        txt_cfg=txt_cfg,
    )
    
    # 无条件（空白图像 + 空白提示）
    uncond_image = load_blank_image(width, height)
    uncond_prompt = "<image>\nGenerate an image."
    input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs(
        model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height, device
    )
    with torch.inference_mode():
        no_both_cond = model.generate_condition(
            input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs
        )

    # 无文本条件（输入图像 + 空白提示）
    input_img_resized = input_img.resize((width, height))
    with torch.inference_mode():
        input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs(
            model, text_tokenizer, visual_tokenizer, uncond_prompt, input_img_resized, width, height, device
        )
        no_txt_cond = model.generate_condition(
            input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs
        )

    # 完整条件（输入图像 + 编辑提示）
    full_prompt = "<image>\n" + prompt.strip()
    input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = build_inputs(
        model, text_tokenizer, visual_tokenizer, full_prompt, input_img_resized, width, height, device
    )
    with torch.inference_mode():
        cond = model.generate_condition(
            input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs
        )
        cond["vae_pixel_values"] = vae_pixel_values
        images = model.generate_img(cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)
    
    return images


def main():
    parser = argparse.ArgumentParser(description='Ovis-U1 Subprocess Worker')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--model-path', required=True, help='Ovis-U1 model path')
    parser.add_argument('--ovis-repo', default='/data2/yixuan/Ovis-U1', help='Ovis-U1 repository path')
    parser.add_argument('--device', default='cuda:0', help='Device (e.g., cuda:0)')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--img-cfg', type=float, default=1.5)
    parser.add_argument('--txt-cfg', type=float, default=6.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable-progress-bar', action='store_true')
    
    args = parser.parse_args()
    
    # 解析设备
    device = args.device
    gpu_id = int(device.split(':')[1]) if ':' in device else 0
    
    print(f"[Worker] Starting on {device}", file=sys.stderr)
    print(f"[Worker] Model Path: {args.model_path}", file=sys.stderr)
    
    try:
        # 添加 Ovis-U1 仓库到 Python 路径（可选，用于自定义代码）
        ovis_repo = Path(args.ovis_repo)
        if ovis_repo.exists() and str(ovis_repo) not in sys.path:
            sys.path.insert(0, str(ovis_repo))
            print(f"[Worker] Added Ovis repo to path: {ovis_repo}", file=sys.stderr)
        
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
        
        # ==================== 加载模型 ====================
        print(f"[Worker] Loading Ovis-U1 model...", file=sys.stderr)
        
        from transformers import AutoModelForCausalLM
        
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            output_loading_info=True,
            trust_remote_code=True
        )
        
        print(f"[Worker] Model loaded, moving to {device}...", file=sys.stderr)
        model = model.eval().to(device)
        model = model.to(torch_dtype)
        
        print(f"[Worker] Model ready on {device}", file=sys.stderr)
        
        # ==================== 处理任务 ====================
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
                
                # 保存原始尺寸
                original_size = image.size
                
                # 执行推理
                print(f"[Worker] Task {task_id}: Running inference ({args.num_inference_steps} steps)...", file=sys.stderr)
                
                edited_images = pipe_img_edit(
                    model=model,
                    input_img=image,
                    prompt=instruction,
                    steps=args.num_inference_steps,
                    txt_cfg=args.txt_cfg,
                    img_cfg=args.img_cfg,
                    seed=current_seed,
                    device=device
                )
                
                # 获取第一张生成的图像
                edited_image = edited_images[0]
                
                # 恢复原始尺寸
                if edited_image.size != original_size:
                    edited_image = edited_image.resize(original_size, Image.LANCZOS)
                
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














