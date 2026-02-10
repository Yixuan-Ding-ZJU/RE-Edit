#!/usr/bin/env python
"""
HiDream-E1.1 Subprocess Worker
在特定虚拟环境中运行 HiDream-E1.1 模型推理

用法:
    python hidream_e1_subprocess_worker.py --input input.json --output output.json \
        --llama-path ... --hidream-i1-path ... --hidream-e1-path ... \
        --hidream-repo ... --device cuda:0
"""

import argparse
import json
import base64
import sys
import math
from io import BytesIO
from pathlib import Path
from collections import defaultdict

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


def resize_image(pil_image, image_size=1024):
    """
    Resize image to fit within image_size x image_size while maintaining aspect ratio
    (Copied from HiDream-E1 inference code)
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    m = 16
    width, height = pil_image.width, pil_image.height
    S_max = image_size * image_size
    scale = S_max / (width * height)
    scale = math.sqrt(scale)

    new_sizes = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    new_sizes = sorted(new_sizes, key=lambda x: x[0] * x[1], reverse=True)

    for new_size in new_sizes:
        if new_size[0] * new_size[1] <= S_max:
            break

    s1 = width / new_size[0]
    s2 = height / new_size[1]
    if s1 < s2:
        pil_image = pil_image.resize([new_size[0], round(height / s1)], resample=Image.BICUBIC)
        top = (round(height / s1) - new_size[1]) // 2
        pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
    else:
        pil_image = pil_image.resize([round(width / s2), new_size[1]], resample=Image.BICUBIC)
        left = (round(width / s2) - new_size[0]) // 2
        pil_image = pil_image.crop((left, 0, left + new_size[0], new_size[1]))

    return pil_image


def load_safetensors(directory):
    """Load sharded safetensors from directory"""
    from safetensors.torch import safe_open
    
    index_file = Path(directory) / "diffusion_pytorch_model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
    
    shards = defaultdict(list)
    for name, file in weight_map.items():
        shards[file].append(name)
    
    state_dict = {}
    for file, names in shards.items():
        file_path = Path(directory) / file
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            state_dict.update({name: f.get_tensor(name) for name in names})
    return state_dict


def main():
    parser = argparse.ArgumentParser(description='HiDream-E1.1 Subprocess Worker')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--llama-path', required=True, help='LLaMA model path')
    parser.add_argument('--hidream-i1-path', required=True, help='HiDream-I1 model path')
    parser.add_argument('--hidream-e1-path', required=True, help='HiDream-E1 weights path')
    parser.add_argument('--hidream-repo', required=True, help='HiDream-E1 repository path')
    parser.add_argument('--device', default='cuda:0', help='Device (e.g., cuda:0)')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--num-inference-steps', type=int, default=28)
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--img-guidance-scale', type=float, default=1.5)
    parser.add_argument('--refine-strength', type=float, default=0.3)
    parser.add_argument('--negative-prompt', default="low quality, blurry, distorted")
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--clip-cfg-norm', action='store_true')
    parser.add_argument('--disable-progress-bar', action='store_true')
    
    args = parser.parse_args()
    
    # 解析设备
    device = args.device
    gpu_id = int(device.split(':')[1]) if ':' in device else 0
    
    print(f"[Worker] Starting on {device}", file=sys.stderr)
    print(f"[Worker] HiDream-E1 Repo: {args.hidream_repo}", file=sys.stderr)
    
    try:
        # 添加 HiDream-E1 仓库到 Python 路径
        hidream_repo = Path(args.hidream_repo)
        if str(hidream_repo) not in sys.path:
            sys.path.insert(0, str(hidream_repo))
        
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
        print(f"[Worker] Loading models...", file=sys.stderr)
        
        # 加载 tokenizer 和 text encoder
        from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
        
        print(f"[Worker] Loading LLaMA tokenizer and text encoder...", file=sys.stderr)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.llama_path)
        text_encoder = LlamaForCausalLM.from_pretrained(
            args.llama_path, 
            output_hidden_states=True, 
            output_attentions=True, 
            torch_dtype=torch_dtype
        )
        
        # 加载 transformer
        from diffusers import HiDreamImageTransformer2DModel
        
        print(f"[Worker] Loading HiDream transformer...", file=sys.stderr)
        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            args.hidream_i1_path, 
            subfolder="transformer"
        )
        transformer.max_seq = 8192
        
        # 加载编辑权重
        print(f"[Worker] Loading edit weights from {args.hidream_e1_path}...", file=sys.stderr)
        src_dict = transformer.state_dict()
        edit_dict = load_safetensors(args.hidream_e1_path + "/transformer")
        reload_keys = {"editing": src_dict, "refine": edit_dict}
        transformer.load_state_dict(edit_dict, strict=True)
        
        # 创建 pipeline
        from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
        
        print(f"[Worker] Creating HiDream pipeline...", file=sys.stderr)
        pipe = HiDreamImageEditingPipeline.from_pretrained(
            args.hidream_i1_path, 
            tokenizer_4=tokenizer, 
            text_encoder_4=text_encoder,
            torch_dtype=torch_dtype, 
            transformer=transformer
        ).to(device, torch_dtype)
        
        print(f"[Worker] Models loaded successfully", file=sys.stderr)
        
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
                
                # 预处理图像（resize）
                processed_image = resize_image(image)
                
                # 准备生成器
                generator = torch.Generator(device=device).manual_seed(current_seed)
                
                # 执行推理
                print(f"[Worker] Task {task_id}: Running inference ({args.num_inference_steps} steps)...", file=sys.stderr)
                
                result = pipe(
                    prompt=instruction,
                    negative_prompt=args.negative_prompt,
                    image=processed_image,
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.img_guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    refine_strength=args.refine_strength,
                    reload_keys=reload_keys,
                    clip_cfg_norm=args.clip_cfg_norm,
                )
                
                # 恢复原始尺寸
                edited_image = result.images[0].resize(original_size)
                
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














