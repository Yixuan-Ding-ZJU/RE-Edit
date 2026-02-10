"""
Qwen-Image-Edit-2511 diffusion model implementation
Qwen图像编辑模型2511版本实现（支持多GPU并行 - 多进程版本）

基于官方 Qwen-Image-Edit-2511 模型（QwenImageEditPlusPipeline）

多进程架构：
- 每个GPU对应一个独立进程
- 进程间完全隔离，避免GIL和资源竞争
- 使用Queue进行进程间通信

================================================================================
⚠️  新增扩散模型实现时的重要注意事项 (CRITICAL FOR NEW IMPLEMENTATIONS)
================================================================================

Pipeline会在多轮评测中复用模型实例，必须正确实现生命周期管理，否则会导致卡死！

【问题场景】
在iterative_pipeline中，多个类别会依次处理：
  第1轮(物理): _initialize() → batch_edit() → unload_from_gpu()
  第2轮(环境): load_to_gpu() → batch_edit() → unload_from_gpu()
  第3轮(社会): load_to_gpu() → batch_edit() → unload_from_gpu()
  ...

【核心问题】
  对象实例复用 ≠ 底层资源（进程/线程/模型）存活
  
  错误做法：
    def load_to_gpu(self):
        print("Model loaded")  # ❌ 只打印日志，进程已死但未重启
        return
  
  结果：第2轮调用batch_edit()时，队列/进程已失效 → 永久卡死

【正确实现】

1. **Multiprocessing模式**（本文件采用的方式）：
   
   def load_to_gpu(self):
       # ✅ 检查进程健康状态
       if hasattr(self, 'processes') and len(self.processes) > 0:
           alive = [p for _, p in self.processes if p.is_alive()]
           if len(alive) < len(self.processes):  # 发现死进程
               self._cleanup_processes()  # 清理死进程和队列
               self._initialize()  # 重新启动所有进程
       else:
           self._initialize()  # 首次加载
   
   def unload_from_gpu(self):
       # ✅ 完全终止所有进程（优雅退出 → terminate → kill）
       for task_queue in self.task_queues:
           task_queue.put(None)  # 发送退出信号
       for gpu_id, p in self.processes:
           p.join(timeout=5)
           if p.is_alive():
               p.terminate()
               p.join(timeout=3)
               if p.is_alive():
                   p.kill()

================================================================================
"""

import multiprocessing as mp
import torch
from PIL import Image
from typing import List, Dict, Any
from tqdm import tqdm
import base64
from io import BytesIO
import sys

from ..base_diffusion import BaseDiffusionModel

# 必须设置，否则多进程会出错
mp.set_start_method('spawn', force=True)


def _load_model_in_process(gpu_id: int, model_name: str, config: Dict[str, Any]):
    """
    在独立进程中加载Qwen-Image-Edit-2511模型
    
    Args:
        gpu_id: GPU ID
        model_name: 模型名称或路径
        config: 模型配置参数
    
    Returns:
        pipeline对象
    """
    print(f"[GPU {gpu_id}] 🔄 Loading Qwen-Image-Edit-2511 model...")
    try:
        from diffusers import QwenImageEditPlusPipeline
        
        # 设置当前设备
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        # 清空GPU缓存
        print(f"[GPU {gpu_id}] 🧹 Clearing GPU cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 解析dtype
        dtype = config.get("dtype", "bfloat16")
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # 加载模型
        print(f"[GPU {gpu_id}] 🔹 Loading Qwen-Image-Edit-2511 pipeline...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        
        # 设置设备
        pipeline.to(device)
        
        # 禁用进度条（可选）
        if config.get("disable_progress_bar", True):
            pipeline.set_progress_bar_config(disable=True)
        else:
            pipeline.set_progress_bar_config(disable=None)
        
        print(f"[GPU {gpu_id}] ✅ Model loaded successfully")
        return pipeline
        
    except Exception as e:
        print(f"[GPU {gpu_id}] ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def _image_to_base64(image: Image.Image) -> str:
    """将PIL Image转换为base64字符串"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _base64_to_image(b64_str: str) -> Image.Image:
    """将base64字符串转换为PIL Image"""
    image_data = base64.b64decode(b64_str)
    return Image.open(BytesIO(image_data))


def _process_worker(gpu_id: int, model_name: str, config: Dict[str, Any], 
                    task_queue: mp.Queue, result_queue: mp.Queue):
    """
    进程工作函数：在独立进程中处理图像编辑任务
    
    Args:
        gpu_id: GPU ID
        model_name: 模型名称或路径
        config: 模型配置参数
        task_queue: 任务队列（接收任务）
        result_queue: 结果队列（发送结果）
    """
    try:
        # 加载模型
        pipeline = _load_model_in_process(gpu_id, model_name, config)
        if pipeline is None:
            print(f"[GPU {gpu_id}] ❌ Failed to load model, exiting...")
            return
        
        # 提取配置参数
        num_inference_steps = config.get("num_inference_steps", 40)
        true_cfg_scale = config.get("true_cfg_scale", 4.0)
        guidance_scale = config.get("guidance_scale", 1.0)
        num_images_per_prompt = config.get("num_images_per_prompt", 1)
        negative_prompt = config.get("negative_prompt", " ")
        seed = config.get("seed", 0)
        
        print(f"[GPU {gpu_id}] ✅ Worker ready, waiting for tasks...")
        
        # 处理任务循环
        while True:
            # 从任务队列获取任务
            task = task_queue.get()
            
            # 检查结束信号
            if task is None:
                print(f"[GPU {gpu_id}] 🛑 Received stop signal, exiting...")
                break
            
            task_id, image_b64, instruction, current_seed, kwargs = task
            
            try:
                # 解码图像
                image = _base64_to_image(image_b64)
                
                # 确保图像是RGB格式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 确保使用正确的设备
                torch.cuda.set_device(gpu_id)
                
                # 准备参数
                num_steps = kwargs.get("num_inference_steps", num_inference_steps)
                cfg_scale = kwargs.get("true_cfg_scale", true_cfg_scale)
                guid_scale = kwargs.get("guidance_scale", guidance_scale)
                num_images = kwargs.get("num_images_per_prompt", num_images_per_prompt)
                neg_prompt = kwargs.get("negative_prompt", negative_prompt)
                use_seed = current_seed if current_seed is not None else seed
                show_progress = kwargs.get("show_progress", True)  # 默认显示进度条
                
                # 准备pipeline输入
                # 注意：Qwen-Image-Edit-2511 使用 torch.Generator，不需要指定device
                generator = torch.Generator()
                generator.manual_seed(use_seed)
                
                # QwenImageEditPlusPipeline 要求 image 是列表格式
                pipeline_inputs = {
                    "image": [image],  # 注意：必须是列表
                    "prompt": instruction,
                    "generator": generator,
                    "true_cfg_scale": cfg_scale,
                    "negative_prompt": neg_prompt,
                    "num_inference_steps": num_steps,
                    "guidance_scale": guid_scale,
                    "num_images_per_prompt": num_images,
                }
                
                # 添加去噪进度条（如果启用）
                pbar = None
                if show_progress:
                    # 为每个GPU进程创建独立的进度条
                    pbar = tqdm(
                        total=num_steps,
                        desc=f"[GPU {gpu_id}] Task {task_id} Denoising",
                        unit="step",
                        leave=False,
                        file=sys.stdout,
                        ncols=100,
                        dynamic_ncols=False,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    )
                    
                    def callback(pipe, step_index, timestep, callback_kwargs):
                        if pbar is not None:
                            pbar.update(1)
                        return callback_kwargs
                    
                    # 尝试添加回调（如果pipeline支持）
                    try:
                        pipeline_inputs["callback_on_step_end"] = callback
                    except:
                        # 如果pipeline不支持callback，则使用pipeline自带的进度条
                        pass
                
                # 执行推理
                try:
                    with torch.inference_mode():
                        output = pipeline(**pipeline_inputs)
                        # 取第一张生成的图像
                        edited_image = output.images[0]
                finally:
                    # 关闭进度条
                    if pbar is not None:
                        pbar.close()
                
                # 编码结果图像
                result_b64 = _image_to_base64(edited_image)
                
                # 发送结果
                result_queue.put((task_id, True, result_b64, None))
                
            except Exception as e:
                print(f"[GPU {gpu_id}] ❌ Error processing task {task_id}: {e}")
                import traceback
                traceback.print_exc()
                # 发送错误结果
                result_queue.put((task_id, False, None, str(e)))
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
        
        print(f"[GPU {gpu_id}] 👋 Worker exiting...")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] ❌ Fatal error in worker: {e}")
        import traceback
        traceback.print_exc()


class QwenImageEdit2511Model(BaseDiffusionModel):
    """
    Qwen-Image-Edit-2511 扩散编辑模型实现（多GPU并行 - 多进程版本）
    
    官方仓库: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
    
    使用multiprocessing实现数据并行：
    - 每个GPU对应一个独立进程
    - 每个进程加载一个完整的模型副本
    - 任务按轮询方式分配到各个GPU进程
    - 所有GPU进程并行处理不同的图像
    
    特点：
    - 支持批次同步，确保GPU间进度一致
    - 进程间完全隔离，避免GIL和资源竞争
    - 使用 QwenImageEditPlusPipeline（2511版本）
    """
    
    def _initialize(self):
        """初始化多GPU模型（多进程版本）"""
        # 获取配置
        self.model_name = self.config.get("model_name", "Qwen/Qwen-Image-Edit-2511")
        device_ids = self.config.get("device_ids", None)
        
        # 确定使用哪些GPU
        if device_ids is None:
            # 如果没有指定，使用所有可用GPU
            self.num_gpus = torch.cuda.device_count()
            self.device_ids = list(range(self.num_gpus))
        else:
            self.device_ids = device_ids
            self.num_gpus = len(device_ids)
        
        print(f"[QwenImageEdit2511] 检测到 {torch.cuda.device_count()} 个GPU")
        print(f"[QwenImageEdit2511] 将使用 {self.num_gpus} 个GPU: {self.device_ids}")
        print(f"[QwenImageEdit2511] 使用多进程架构（每个GPU一个独立进程）\n")
        
        # 创建进程间通信队列
        self.task_queues = [mp.Queue() for _ in range(self.num_gpus)]
        self.result_queue = mp.Queue()
        
        # 启动工作进程
        self.processes = []
        print("=" * 70)
        print("🚀 Starting Worker Processes (Qwen-Image-Edit-2511)")
        print("=" * 70)
        print(f"Starting {self.num_gpus} worker processes...")
        print("(Each process will load model independently)")
        print()
        
        for idx, gpu_id in enumerate(self.device_ids):
            print(f"[{idx+1}/{self.num_gpus}] Starting process for GPU {gpu_id}...")
            
            p = mp.Process(
                target=_process_worker,
                args=(
                    gpu_id,
                    self.model_name,
                    self.config,
                    self.task_queues[idx],
                    self.result_queue
                ),
                name=f"GPU-{gpu_id}"
            )
            p.start()
            self.processes.append((gpu_id, p))
            print(f"  ✅ GPU {gpu_id}: Process started (PID: {p.pid})\n")
        
        print(f"✅ Successfully started {len(self.processes)} worker processes")
        print(f"  ⚡ All processes are loading models independently")
        print("=" * 70)
        print()
    
    def edit_image(self, 
                   original_image: Image.Image,
                   edit_instruction: str,
                   **kwargs) -> Image.Image:
        """
        编辑单张图像（使用第一个GPU进程）
        
        Args:
            original_image: 原始PIL图像
            edit_instruction: 编辑指令文本
            **kwargs: 其他参数
            
        Returns:
            编辑后的PIL图像
        """
        # 单张图像使用第一个GPU进程
        results = self.batch_edit([original_image], [edit_instruction], **kwargs)
        return results[0]
    
    def batch_edit(self,
                   images: List[Image.Image],
                   instructions: List[str],
                   **kwargs) -> List[Image.Image]:
        """
        多GPU并行批量编辑图像（带批次同步）
        
        实现批次同步机制：
        - 将任务分成多个批次，每批大小 = GPU数量
        - 每批任务提交后，等待所有GPU完成
        - 再提交下一批，确保GPU之间保持同步
        
        Args:
            images: 原始图像列表
            instructions: 编辑指令列表
            **kwargs: 其他参数
                - enable_batch_sync: 是否启用批次同步（默认True）
                - num_inference_steps: 推理步数
                - true_cfg_scale: CFG scale
                - guidance_scale: Guidance scale（默认1.0）
                - num_images_per_prompt: 每个prompt生成的图像数（默认1）
                - negative_prompt: 负面提示词
                - seed: 随机种子
                - show_progress: 是否显示进度条（默认True）
            
        Returns:
            编辑后的图像列表
        """
        if len(images) != len(instructions):
            raise ValueError("Number of images must match number of instructions")
        
        n = len(images)
        num_gpus = self.num_gpus
        enable_sync = kwargs.pop("enable_batch_sync", True)  # 默认启用批次同步
        
        print(f"\n[QwenImageEdit2511] Starting batch edit: {n} images on {num_gpus} GPUs")
        print(f"  🔄 Batch synchronization: {'ENABLED ✅' if enable_sync else 'DISABLED ⚠️'}")
        
        # 预先分配任务并显示
        print("=" * 70)
        print("📋 Task Assignment:")
        print("=" * 70)
        from collections import defaultdict
        gpu_assignments = defaultdict(list)
        
        for idx in range(n):
            gpu_id = self.device_ids[idx % num_gpus]
            gpu_assignments[gpu_id].append(idx)
        
        for gpu_id in sorted(gpu_assignments.keys()):
            assigned = gpu_assignments[gpu_id]
            print(f"  GPU {gpu_id}: {len(assigned)} images")
            preview = ", ".join(map(str, assigned[:5]))
            if len(assigned) > 5:
                preview += f", ... +{len(assigned) - 5} more"
            print(f"           → [{preview}]")
        
        print("=" * 70)
        print()
        
        # 结果列表（保持原始顺序）
        results = [None] * n
        
        # 获取基础seed
        base_seed = kwargs.get("seed", self.config.get("seed", 0))
        
        if enable_sync:
            # 批次同步模式：每批num_gpus个任务，批次间同步
            results = self._batch_edit_with_sync(
                images, instructions, n, num_gpus, base_seed, **kwargs
            )
        else:
            # 原始模式：一次性提交所有任务（向后兼容）
            results = self._batch_edit_no_sync(
                images, instructions, n, num_gpus, base_seed, **kwargs
            )
        
        print(f"✅ Batch edit completed: {n} images\n")
        return results
    
    def _batch_edit_with_sync(self, images, instructions, n, num_gpus, base_seed, **kwargs):
        """
        批次同步模式：确保每批所有GPU完成后再开始下一批（多进程版本）
        """
        results = [None] * n
        
        # 计算批次数
        num_batches = (n + num_gpus - 1) // num_gpus
        
        print(f"🔄 Batch synchronization mode (multiprocess):")
        print(f"   - Total batches: {num_batches}")
        print(f"   - Batch size: {num_gpus} (one task per GPU process)")
        print(f"   - All GPU processes will stay synchronized at batch boundaries\n")
        
        # 总进度条
        with tqdm(total=n, desc="[SYNC] Editing images", unit="img") as pbar:
            # 逐批处理
            for batch_idx in range(num_batches):
                batch_start = batch_idx * num_gpus
                batch_end = min(batch_start + num_gpus, n)
                batch_size = batch_end - batch_start
                
                # 准备当前批次的任务
                task_indices = []
                for i in range(batch_start, batch_end):
                    # 使用全局轮询分配：i % num_gpus 确保所有任务按顺序轮询分配到不同GPU
                    gpu_idx = i % num_gpus
                    current_seed = base_seed + i
                    
                    # 编码图像
                    image_b64 = _image_to_base64(images[i])
                    
                    # 发送任务到对应的GPU进程
                    task = (i, image_b64, instructions[i], current_seed, kwargs)
                    self.task_queues[gpu_idx].put(task)
                    task_indices.append(i)
                
                # 等待当前批次所有任务完成（同步点）
                # 使用字典跟踪当前批次的任务，确保只收集当前批次的结果
                batch_results = {}
                expected_task_ids = set(task_indices)
                
                # 等待直到收集到当前批次的所有结果
                # 这是真正的同步点：必须等待当前批次的所有任务完成才能继续下一批
                while len(batch_results) < len(task_indices):
                    try:
                        task_id, success, result_b64, error = self.result_queue.get()
                        
                        # 只处理属于当前批次的任务结果
                        if task_id in expected_task_ids:
                            if success:
                                batch_results[task_id] = _base64_to_image(result_b64)
                            else:
                                print(f"\n❌ Error editing image {task_id}: {error}")
                                batch_results[task_id] = images[task_id]  # fallback
                            pbar.update(1)
                        else:
                            # 如果收到不属于当前批次的结果，说明有进程提前完成了任务
                            # 这种情况不应该在同步模式下发生（因为任务是按批次提交的）
                            # 但为了健壮性，我们仍然处理它，但会记录警告
                            print(f"\n⚠️  [SYNC] Received result for task {task_id} outside current batch {expected_task_ids}")
                            # 直接处理，因为task_id可以正确匹配到results数组
                            if success:
                                results[task_id] = _base64_to_image(result_b64)
                            else:
                                results[task_id] = images[task_id]
                            pbar.update(1)
                    except Exception as e:
                        print(f"\n❌ Error receiving result: {e}")
                
                # 将当前批次的结果写入results数组（确保所有结果都已收集）
                for task_id in task_indices:
                    if task_id in batch_results:
                        results[task_id] = batch_results[task_id]
                    else:
                        # 如果某个任务没有结果（理论上不应该发生），使用原图作为fallback
                        print(f"\n⚠️  [SYNC] Missing result for task {task_id} in batch {batch_idx}")
                        results[task_id] = images[task_id]
                
                # 当前批次完成，所有GPU进程已同步，可以开始下一批
                if batch_idx < num_batches - 1:
                    pbar.set_postfix_str(f"Batch {batch_idx+1}/{num_batches} done, GPUs synced ✓")
        
        return results
    
    def _batch_edit_no_sync(self, images, instructions, n, num_gpus, base_seed, **kwargs):
        """
        无同步模式：一次性提交所有任务（多进程版本）
        """
        results = [None] * n
        
        print(f"⚡ No-sync mode (multiprocess): All {n} tasks submitted at once\n")
        
        # 提交所有任务
        for idx in range(n):
            gpu_idx = idx % num_gpus
            current_seed = base_seed + idx
            
            # 编码图像
            image_b64 = _image_to_base64(images[idx])
            
            # 发送任务到对应的GPU进程
            task = (idx, image_b64, instructions[idx], current_seed, kwargs)
            self.task_queues[gpu_idx].put(task)
        
        # 收集结果（带进度条）
        with tqdm(total=n, desc="[NO-SYNC] Editing images", unit="img") as pbar:
            for _ in range(n):
                try:
                    task_id, success, result_b64, error = self.result_queue.get()
                    
                    if success:
                        results[task_id] = _base64_to_image(result_b64)
                    else:
                        print(f"\n❌ Error editing image {task_id}: {error}")
                        results[task_id] = images[task_id]
                except Exception as e:
                    print(f"\n❌ Error receiving result: {e}")
                finally:
                    pbar.update(1)
        
        return results
    
    def _cleanup_processes(self):
        """
        清理所有worker进程和队列
        """
        if not hasattr(self, 'processes'):
            return
        
        print(f"[QwenImageEdit2511] 🧹 Cleaning up {len(self.processes)} worker processes...")
        
        # 向所有存活的进程发送停止信号
        if hasattr(self, 'task_queues'):
            for task_queue in self.task_queues:
                try:
                    task_queue.put(None)
                except:
                    pass  # 队列可能已损坏
        
        # 终止所有进程
        for gpu_id, p in self.processes:
            if p.is_alive():
                try:
                    p.terminate()
                    p.join(timeout=5)
                    print(f"[QwenImageEdit2511]   ✅ GPU {gpu_id} process terminated")
                except Exception as e:
                    print(f"[QwenImageEdit2511]   ⚠️  Error terminating GPU {gpu_id} process: {e}")
            else:
                print(f"[QwenImageEdit2511]   ✓ GPU {gpu_id} process already dead")
        
        # 清理队列（重要：防止队列堆积导致内存泄漏）
        if hasattr(self, 'task_queues'):
            for task_queue in self.task_queues:
                # 清空队列中的所有任务
                while not task_queue.empty():
                    try:
                        task_queue.get_nowait()
                    except:
                        break
        
        if hasattr(self, 'result_queue'):
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except:
                    break
        
        print(f"[QwenImageEdit2511] ✅ Cleanup complete")
    
    def unload_from_gpu(self):
        """
        停止所有工作进程（清理资源）
        """
        if not hasattr(self, 'processes') or len(self.processes) == 0:
            print(f"[QwenImageEdit2511] No processes to unload")
            return
        
        print(f"[QwenImageEdit2511] Stopping {len(self.processes)} worker processes...")
        
        # 向所有进程发送停止信号（优雅退出）
        if hasattr(self, 'task_queues'):
            for task_queue in self.task_queues:
                try:
                    task_queue.put(None)
                except:
                    pass  # 队列可能已损坏
        
        # 等待所有进程优雅退出（最多5秒每个进程）
        import time
        start_time = time.time()
        for gpu_id, p in self.processes:
            if not p.is_alive():
                print(f"[QwenImageEdit2511] ✓ GPU {gpu_id} process already stopped")
                continue
            
            try:
                remaining_time = max(1, 5 - (time.time() - start_time))
                p.join(timeout=remaining_time)
                
                if p.is_alive():
                    print(f"[QwenImageEdit2511] ⚠️  GPU {gpu_id} process did not terminate gracefully, forcing...")
                    p.terminate()
                    p.join(timeout=3)
                    
                    # 如果terminate还不行，使用kill
                    if p.is_alive():
                        print(f"[QwenImageEdit2511] ⚠️  GPU {gpu_id} process did not respond to SIGTERM, killing...")
                        p.kill()
                        p.join(timeout=2)
                        
                        if p.is_alive():
                            print(f"[QwenImageEdit2511] ❌ GPU {gpu_id} process is unresponsive (zombie)")
                        else:
                            print(f"[QwenImageEdit2511] ✅ GPU {gpu_id} process killed")
                    else:
                        print(f"[QwenImageEdit2511] ✅ GPU {gpu_id} process terminated")
                else:
                    print(f"[QwenImageEdit2511] ✅ GPU {gpu_id} process stopped gracefully")
                    
            except Exception as e:
                print(f"[QwenImageEdit2511] ⚠️  Error stopping GPU {gpu_id} process: {e}")
        
        # 清理队列中的残留数据（防止内存泄漏）
        if hasattr(self, 'task_queues'):
            for i, task_queue in enumerate(self.task_queues):
                cleared = 0
                while not task_queue.empty():
                    try:
                        task_queue.get_nowait()
                        cleared += 1
                    except:
                        break
                if cleared > 0:
                    print(f"[QwenImageEdit2511] 🧹 Cleared {cleared} pending tasks from GPU {self.device_ids[i]} queue")
        
        if hasattr(self, 'result_queue'):
            cleared = 0
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                    cleared += 1
                except:
                    break
            if cleared > 0:
                print(f"[QwenImageEdit2511] 🧹 Cleared {cleared} pending results from result queue")
        
        print(f"[QwenImageEdit2511] ✅ All worker processes stopped")
    
    def load_to_gpu(self, parallel: bool = True):
        """
        将模型加载到GPU（重新启动worker进程）
        
        多进程版本需要检查进程是否存活，如果进程已死则重新启动
        """
        # 检查进程是否存活
        if hasattr(self, 'processes') and len(self.processes) > 0:
            alive_processes = [p for _, p in self.processes if p.is_alive()]
            dead_processes = [gpu_id for gpu_id, p in self.processes if not p.is_alive()]
            
            if len(alive_processes) == len(self.processes):
                print(f"[QwenImageEdit2511] ✅ All {len(self.processes)} worker processes are already running")
                return
            else:
                print(f"[QwenImageEdit2511] ⚠️  Detected {len(dead_processes)} dead processes: {dead_processes}")
                print(f"[QwenImageEdit2511] 🔄 Restarting all worker processes...")
                # 清理所有进程和队列
                self._cleanup_processes()
        
        # 重新初始化（启动新进程）
        print(f"[QwenImageEdit2511] 🚀 Initializing worker processes...")
        self._initialize()
    
    def __del__(self):
        """清理资源（析构函数）"""
        if hasattr(self, 'processes') and len(self.processes) > 0:
            # 确保所有进程都已停止（使用强制方式）
            for gpu_id, p in self.processes:
                if p.is_alive():
                    try:
                        # 直接terminate，不等待优雅退出
                        p.terminate()
                        p.join(timeout=2)
                        
                        # 如果还活着，强制kill
                        if p.is_alive():
                            p.kill()
                            p.join(timeout=1)
                    except:
                        pass  # 析构函数不应抛出异常






