#!/usr/bin/env python
"""
Janus Subprocess Worker
在特定虚拟环境中运行 Janus 模型推理

用法:
    python janus_subprocess_worker.py --input input.json --output output.json \
        --model-path FreedomIntelligence/Janus-4o-7B --janus-repo /path/to/Janus --device cuda:0
"""

import argparse
import json
import base64
import sys
import os
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
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


@dataclass
class VLChatProcessorOutput:
    """VLChatProcessor 输出数据类"""
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: list

    def __len__(self):
        return len(self.input_ids)


def process_image(images, vl_chat_processor):
    """处理图像列表"""
    images_outputs = vl_chat_processor.image_processor(images, return_tensors="pt")
    return images_outputs['pixel_values']


def text_and_image_to_image_generate(
    input_prompt, 
    input_image, 
    vl_chat_processor, 
    vl_gpt, 
    device,
    temperature=1.0, 
    parallel_size=1, 
    cfg_weight=5.0, 
    cfg_weight2=5.0,
    seed=42
):
    """
    文本+图像到图像生成
    
    Args:
        input_prompt: 编辑指令
        input_image: PIL Image 输入图像
        vl_chat_processor: VLChatProcessor
        vl_gpt: MultiModalityCausalLM
        device: 设备
        temperature: 采样温度
        parallel_size: 并行生成数量
        cfg_weight: CFG 权重
        cfg_weight2: CFG 权重2
        seed: 随机种子
        
    Returns:
        生成的图像列表
    """
    torch.cuda.empty_cache()
    
    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 构建输入token
    input_img_tokens = (
        vl_chat_processor.image_start_tag + 
        vl_chat_processor.image_tag * vl_chat_processor.num_image_tokens +
        vl_chat_processor.image_end_tag + 
        vl_chat_processor.image_start_tag + 
        vl_chat_processor.pad_tag * vl_chat_processor.num_image_tokens +
        vl_chat_processor.image_end_tag
    )
    output_img_tokens = vl_chat_processor.image_start_tag 

    pre_data = []
    input_images = [input_image]  # PIL Image
    img_len = len(input_images)
    prompts = input_img_tokens * img_len + input_prompt
    
    conversation = [
        {"role": "<|User|>", "content": prompts},
        {"role": "<|Assistant|>", "content": ""}
    ]
    
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    sft_format = sft_format + output_img_tokens

    mmgpt = vl_gpt

    image_token_num_per_image = 576
    img_size = 384
    patch_size = 16

    with torch.inference_mode():
        # 处理输入图像
        input_image_pixel_values = process_image(input_images, vl_chat_processor).to(torch.bfloat16).to(device)
        quant_input, emb_loss_input, info_input = mmgpt.gen_vision_model.encode(input_image_pixel_values)
        image_tokens_input = info_input[2].detach().reshape(input_image_pixel_values.shape[0], -1)
        image_embeds_input = mmgpt.prepare_gen_img_embeds(image_tokens_input)

        input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        
        encoder_pixel_values = process_image(input_images, vl_chat_processor).to(device)
        tokens = torch.zeros((parallel_size * 3, len(input_ids)), dtype=torch.long)
        
        for i in range(parallel_size * 3):
            tokens[i, :] = input_ids
            if i % 3 == 2:
                tokens[i, 1:-1] = vl_chat_processor.pad_id
                pre_data.append(VLChatProcessorOutput(
                    sft_format=sft_format, 
                    pixel_values=encoder_pixel_values, 
                    input_ids=tokens[i-2], 
                    num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len
                ))
                pre_data.append(VLChatProcessorOutput(
                    sft_format=sft_format, 
                    pixel_values=encoder_pixel_values, 
                    input_ids=tokens[i-1], 
                    num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len
                ))
                pre_data.append(VLChatProcessorOutput(
                    sft_format=sft_format, 
                    pixel_values=None, 
                    input_ids=tokens[i], 
                    num_image_tokens=[]
                ))

        prepare_inputs = vl_chat_processor.batchify(pre_data)

        inputs_embeds = mmgpt.prepare_inputs_embeds(
            input_ids=tokens.to(device),
            pixel_values=prepare_inputs['pixel_values'].to(torch.bfloat16).to(device),
            images_emb_mask=prepare_inputs['images_emb_mask'].to(device),
            images_seq_mask=prepare_inputs['images_seq_mask'].to(device)
        )

        image_gen_indices = (tokens == vl_chat_processor.image_end_id).nonzero()

        for ii, ind in enumerate(image_gen_indices):
            if ii % 4 == 0:
                offset = ind[1] + 2
                inputs_embeds[ind[0], offset: offset + image_embeds_input.shape[1], :] = image_embeds_input[(ii // 2) % img_len]

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
        
        outputs = None
        for i in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds, 
                use_cache=True, 
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond_full = logits[0::3, :]
            logit_cond_part = logits[1::3, :]
            logit_uncond = logits[2::3, :]

            logit_cond = (logit_cond_full + cfg_weight2 * logit_cond_part) / (1 + cfg_weight2)
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([
                next_token.unsqueeze(dim=1), 
                next_token.unsqueeze(dim=1), 
                next_token.unsqueeze(dim=1)
            ], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = mmgpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int), 
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        output_images = []
        for i in range(parallel_size):
            output_images.append(Image.fromarray(visual_img[i]))
        
        return output_images


def main():
    parser = argparse.ArgumentParser(description='Janus Subprocess Worker')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--model-path', required=True, help='Janus model path')
    parser.add_argument('--janus-repo', required=True, help='Janus repository path')
    parser.add_argument('--device', default='cuda:0', help='Device (e.g., cuda:0)')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--parallel-size', type=int, default=1)
    parser.add_argument('--cfg-weight', type=float, default=5.0)
    parser.add_argument('--cfg-weight2', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable-progress-bar', action='store_true')
    
    args = parser.parse_args()
    
    # 解析设备
    device = args.device
    gpu_id = int(device.split(':')[1]) if ':' in device else 0
    
    print(f"[Worker] Starting on {device}", file=sys.stderr)
    print(f"[Worker] Model Path: {args.model_path}", file=sys.stderr)
    print(f"[Worker] Janus Repo: {args.janus_repo}", file=sys.stderr)
    
    try:
        # 添加 Janus 仓库到 Python 路径
        janus_repo = Path(args.janus_repo)
        if str(janus_repo) not in sys.path:
            sys.path.insert(0, str(janus_repo))
            print(f"[Worker] Added Janus repo to path: {janus_repo}", file=sys.stderr)
        
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
        print(f"[Worker] Loading Janus model...", file=sys.stderr)
        
        from transformers import AutoModelForCausalLM
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        
        # 加载处理器
        vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
        tokenizer = vl_chat_processor.tokenizer
        
        # 加载模型
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        vl_gpt = vl_gpt.to(device).eval()
        
        print(f"[Worker] Model loaded on {device}", file=sys.stderr)
        
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
                print(f"[Worker] Task {task_id}: Running inference...", file=sys.stderr)
                
                edited_images = text_and_image_to_image_generate(
                    input_prompt=instruction,
                    input_image=image,
                    vl_chat_processor=vl_chat_processor,
                    vl_gpt=vl_gpt,
                    device=device,
                    temperature=args.temperature,
                    parallel_size=args.parallel_size,
                    cfg_weight=args.cfg_weight,
                    cfg_weight2=args.cfg_weight2,
                    seed=current_seed
                )
                
                # 获取第一张生成的图像
                edited_image = edited_images[0]
                
                # 恢复原始尺寸（Janus 输出固定 384x384）
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














