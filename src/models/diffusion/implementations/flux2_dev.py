"""
Flux.2-dev diffusion model implementation
Flux.2-dev 图像编辑模型实现（支持多GPU并行 - 多进程版本）

基于 black-forest-labs/FLUX.2-dev
使用 enable_model_cpu_offload() 优化显存使用

多进程架构：
- 每个GPU对应一个独立进程
- 进程间完全隔离，避免GIL和资源竞争
- 使用Queue进行进程间通信
"""

import multiprocessing as mp
import torch
from PIL import Image
from typing import List, Dict, Any
from tqdm import tqdm
import base64
from io import BytesIO
import sys
import threading

from ..base_diffusion import BaseDiffusionModel
from ....utils import setup_logger

# 必须设置，否则多进程会出错
mp.set_start_method('spawn', force=True)


def _load_model_in_process(gpu_id: int, model_name: str, config: Dict[str, Any]):
    """
    在独立进程中加载模型
    
    Args:
        gpu_id: GPU ID
        model_name: 模型名称或路径
        config: 模型配置参数
    
    Returns:
        pipeline对象
    """
    print(f"[GPU {gpu_id}] 🔄 Loading Flux.2-dev model...")
    try:
        from diffusers import Flux2Pipeline
        
        # 设置当前设备
        torch.cuda.set_device(gpu_id)
        
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
        print(f"[GPU {gpu_id}] 🔹 Loading Flux.2-dev pipeline...")
        pipeline = Flux2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        
        # 使用 enable_model_cpu_offload() 优化显存使用
        use_cpu_offload = config.get("use_cpu_offload", True)
        if use_cpu_offload:
            print(f"[GPU {gpu_id}] 🔹 Enabling model CPU offload for GPU {gpu_id}...")
            pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            # 如果不使用CPU offload，直接移动到目标GPU
            print(f"[GPU {gpu_id}] 🔹 Moving model to cuda:{gpu_id}...")
            pipeline.to(f"cuda:{gpu_id}")
        
        # 禁用进度条
        if config.get("disable_progress_bar", True):
            pipeline.set_progress_bar_config(disable=True)
        
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
        num_inference_steps = config.get("num_inference_steps", 50)
        guidance_scale = config.get("guidance_scale", 4.0)
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
                guidance = kwargs.get("guidance_scale", guidance_scale)
                use_seed = current_seed if current_seed is not None else seed
                show_progress = kwargs.get("show_progress", True)  # 默认显示进度条
                
                # 准备pipeline输入
                pipeline_inputs = {
                    "prompt": instruction,
                    "image": [image],
                    "generator": torch.Generator(device=f"cuda:{gpu_id}").manual_seed(use_seed),
                    "guidance_scale": guidance,
                    "num_inference_steps": num_steps,
                }
                
                # 添加去噪进度条（如果启用）
                pbar = None
                if show_progress:
                    # 为每个GPU进程创建独立的进度条
                    # 在多进程环境下，每个进程会独立显示自己的进度条
                    # 使用GPU ID和Task ID作为标识，便于区分不同进程的进度
                    pbar = tqdm(
                        total=num_steps,
                        desc=f"[GPU {gpu_id}] Task {task_id} Denoising",
                        unit="step",
                        leave=False,  # 完成后清除，避免输出混乱
                        file=sys.stdout,
                        ncols=100,  # 限制宽度，避免多进程输出时混乱
                        dynamic_ncols=False,  # 固定宽度
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    )
                    
                    def callback(pipe, step_index, timestep, callback_kwargs):
                        if pbar is not None:
                            pbar.update(1)
                        return callback_kwargs
                    
                    pipeline_inputs["callback_on_step_end"] = callback
                
                # 执行推理
                try:
                    with torch.inference_mode():
                        output = pipeline(**pipeline_inputs)
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
    


class Flux2DevModel(BaseDiffusionModel):
    """
    Flux.2-dev 图像编辑模型（多GPU并行 - 多进程版本）
    
    使用multiprocessing实现数据并行：
    - 每个GPU对应一个独立进程
    - 每个进程加载一个完整的模型副本（使用CPU offload优化显存）
    - 任务按轮询方式分配到各个GPU进程
    - 所有GPU进程并行处理不同的图像
    
    特点：
    - 使用 enable_model_cpu_offload() 自动管理显存
    - 支持批次同步，确保GPU间进度一致
    - 进程间完全隔离，避免GIL和资源竞争
    """
    
    def _initialize(self):
        """初始化多GPU模型（多进程版本）"""
        # 获取配置
        self.model_name = self.config.get("model_name", "black-forest-labs/FLUX.2-dev")
        device_ids = self.config.get("device_ids", None)
        
        # 确定使用哪些GPU
        if device_ids is None:
            # 如果没有指定，使用所有可用GPU
            self.num_gpus = torch.cuda.device_count()
            self.device_ids = list(range(self.num_gpus))
        else:
            self.device_ids = device_ids
            self.num_gpus = len(device_ids)
        
        print(f"[Flux2Dev] 检测到 {torch.cuda.device_count()} 个GPU")
        print(f"[Flux2Dev] 将使用 {self.num_gpus} 个GPU: {self.device_ids}")
        print(f"[Flux2Dev] 使用多进程架构（每个GPU一个独立进程）\n")
        
        # 创建进程间通信队列
        self.task_queues = [mp.Queue() for _ in range(self.num_gpus)]
        self.result_queue = mp.Queue()
        
        # 用于迭代refinement模式的结果分发（线程安全）
        # key: task_id, value: (result_data, event)
        self._pending_results = {}  # 存储待分发的结果
        self._result_lock = threading.Lock()  # 保护pending_results的锁
        self._result_dispatcher_thread = None  # 结果分发线程
        self._result_dispatcher_started = False  # 标记ResultDispatcher是否已启动
        
        # 启动工作进程
        self.processes = []
        print("=" * 70)
        print("🚀 Starting Worker Processes (Flux.2-dev)")
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
    
    def _start_result_dispatcher(self):
        """
        启动结果分发线程（仅在迭代refinement模式下使用）
        
        该线程从result_queue中读取结果，并将其分发到对应的等待线程
        用于支持迭代refinement模式下的多线程并发
        """
        # 如果已经启动，直接返回
        if self._result_dispatcher_started:
            print(f"[Flux2Dev] ResultDispatcher already started, skipping")
            return
        
        def dispatcher():
            """结果分发器：从result_queue读取结果并分发到对应的等待线程（仅用于迭代refinement模式）"""
            print(f"[Flux2Dev] [ResultDispatcher] Thread started")
            try:
                while True:
                    try:
                        # 从共享结果队列读取结果
                        result_data = self.result_queue.get()
                        if result_data is None:  # 退出信号
                            print(f"[Flux2Dev] [ResultDispatcher] Received exit signal, stopping...")
                            break
                        
                        task_id, success, result_b64, error = result_data
                        
                        # 将结果存储到pending_results，并通知等待的线程
                        # 注意：ResultDispatcher只在迭代模式下运行，所以所有结果都应该在_pending_results中
                        with self._result_lock:
                            if task_id in self._pending_results:
                                # 找到等待该结果的线程（迭代refinement模式）
                                event, result_container = self._pending_results[task_id]
                                result_container['data'] = (success, result_b64, error)
                                event.set()  # 通知等待的线程
                            else:
                                # 这种情况不应该发生（因为ResultDispatcher只在迭代模式下运行）
                                # 如果发生了，说明有bug，记录警告但不处理（让batch_edit自己处理）
                                print(f"⚠️  [Flux2Dev] [ResultDispatcher] Received result for task {task_id} but no waiting thread found (this should not happen in iterative mode)")
                    
                    except Exception as e:
                        print(f"❌ [Flux2Dev] [ResultDispatcher] Error in dispatcher loop: {e}")
                        import traceback
                        traceback.print_exc()
                        # 继续运行，不要因为单个错误而停止
            except Exception as e:
                print(f"❌ [Flux2Dev] [ResultDispatcher] Fatal error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print(f"[Flux2Dev] [ResultDispatcher] Thread stopped")
        
        self._result_dispatcher_thread = threading.Thread(target=dispatcher, daemon=True, name="ResultDispatcher")
        self._result_dispatcher_thread.start()
        self._result_dispatcher_started = True
        print(f"[Flux2Dev] ResultDispatcher thread started (thread_id: {self._result_dispatcher_thread.ident})")
    
    def _stop_result_dispatcher(self):
        """停止结果分发线程"""
        if not self._result_dispatcher_started:
            return
        
        if self._result_dispatcher_thread and self._result_dispatcher_thread.is_alive():
            print(f"[Flux2Dev] Stopping ResultDispatcher thread...")
            # 发送退出信号
            try:
                self.result_queue.put(None)
            except Exception as e:
                print(f"[Flux2Dev] Warning: Failed to send exit signal to ResultDispatcher: {e}")
            
            # 等待线程退出
            self._result_dispatcher_thread.join(timeout=2)
            if self._result_dispatcher_thread.is_alive():
                print(f"[Flux2Dev] Warning: ResultDispatcher thread did not stop within timeout")
            else:
                print(f"[Flux2Dev] ResultDispatcher thread stopped successfully")
        
        self._result_dispatcher_thread = None
        self._result_dispatcher_started = False
    
    def edit_image(self, original_image: Image.Image, 
                   edit_instruction: str,
                   **kwargs) -> Image.Image:
        """
        编辑单张图像（使用第一个GPU进程）
        
        Args:
            original_image: 原始PIL图像
            edit_instruction: 编辑指令
            **kwargs: 其他参数
                - target_gpu_id: 指定目标GPU ID（用于迭代refinement模式）
                - enable_batch_sync: 是否启用批次同步（默认True）
            
        Returns:
            编辑后的PIL图像
        """
        # 如果指定了target_gpu_id，直接使用该GPU（用于迭代refinement模式）
        target_gpu_id = kwargs.pop("target_gpu_id", None)
        if target_gpu_id is not None:
            return self._edit_image_on_specific_gpu(original_image, edit_instruction, target_gpu_id, **kwargs)
        
        # 单张图像使用第一个GPU进程（默认行为）
        results = self.batch_edit([original_image], [edit_instruction], **kwargs)
        return results[0]
    
    def _edit_image_on_specific_gpu(self, original_image: Image.Image, edit_instruction: str, 
                                     target_gpu_id: int, **kwargs) -> Image.Image:
        """
        在指定GPU上编辑单张图像（用于迭代refinement模式，避免batch_sync）
        
        Args:
            original_image: 原始PIL图像
            edit_instruction: 编辑指令文本
            target_gpu_id: 目标GPU ID
            **kwargs: 其他参数
            
        Returns:
            编辑后的PIL图像
        """
        # 确保ResultDispatcher已启动（迭代模式需要）
        if not self._result_dispatcher_started:
            self._start_result_dispatcher()
        
        # 验证ResultDispatcher线程确实在运行
        if not (self._result_dispatcher_thread and self._result_dispatcher_thread.is_alive()):
            raise RuntimeError(f"ResultDispatcher thread is not running! This is required for iterative refinement mode.")
        
        # 找到target_gpu_id对应的队列索引
        if target_gpu_id not in self.device_ids:
            raise ValueError(f"Target GPU {target_gpu_id} not in device_ids {self.device_ids}")
        
        gpu_idx = self.device_ids.index(target_gpu_id)
        
        # 生成唯一的任务ID（使用时间戳+随机数+GPU ID避免冲突）
        import time
        import random
        # 使用微秒时间戳 + GPU ID + 随机数，确保全局唯一
        task_id = int(time.time() * 1000000) + target_gpu_id * 10000 + random.randint(0, 9999)
        
        # 准备参数
        base_seed = kwargs.get("seed", self.config.get("seed", 0))
        current_seed = base_seed + task_id
        
        # 编码图像
        image_b64 = _image_to_base64(original_image)
        
        # 创建等待事件和结果容器（线程安全）
        result_event = threading.Event()
        result_container = {'data': None}
        
        # 注册等待该任务的结果
        with self._result_lock:
            self._pending_results[task_id] = (result_event, result_container)
        
        try:
            # 发送任务到指定的GPU队列
            task = (task_id, image_b64, edit_instruction, current_seed, kwargs)
            self.task_queues[gpu_idx].put(task)
            
            # 等待结果（使用事件机制，避免从共享队列错误读取）
            if result_event.wait(timeout=300):  # 5分钟超时
                # 获取结果
                with self._result_lock:
                    if task_id in self._pending_results:
                        del self._pending_results[task_id]
                    success, result_b64, error = result_container['data']
                
                if success:
                    return _base64_to_image(result_b64)
                else:
                    raise RuntimeError(f"Error editing image on GPU {target_gpu_id}: {error}")
            else:
                raise RuntimeError(f"Timeout waiting for result from GPU {target_gpu_id} (task_id: {task_id})")
        except Exception as e:
            # 清理pending_results
            with self._result_lock:
                if task_id in self._pending_results:
                    del self._pending_results[task_id]
            raise RuntimeError(f"Error receiving result from GPU {target_gpu_id}: {e}")
    
    def batch_edit(self, images: List[Image.Image],
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
                - guidance_scale: Guidance scale
                - negative_prompt: 负面提示词
            
        Returns:
            编辑后的图像列表
        """
        # 确保ResultDispatcher未启动（batch_edit模式不需要，直接使用队列）
        if self._result_dispatcher_started:
            self._stop_result_dispatcher()
        if len(images) != len(instructions):
            raise ValueError("Number of images must match number of instructions")
        
        n = len(images)
        num_gpus = self.num_gpus  # 使用进程数量而不是workers数量
        enable_sync = kwargs.pop("enable_batch_sync", True)  # 默认启用批次同步
        
        print(f"\n[Flux2Dev] Starting batch edit: {n} images on {num_gpus} GPUs")
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
        # 停止结果分发线程
        if hasattr(self, '_result_dispatcher_thread'):
            self._stop_result_dispatcher()
        
        if not hasattr(self, 'processes'):
            return
        
        print(f"[Flux2Dev] 🧹 Cleaning up {len(self.processes)} worker processes...")
        
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
                    print(f"[Flux2Dev]   ✅ GPU {gpu_id} process terminated")
                except Exception as e:
                    print(f"[Flux2Dev]   ⚠️  Error terminating GPU {gpu_id} process: {e}")
            else:
                print(f"[Flux2Dev]   ✓ GPU {gpu_id} process already dead")
        
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
        
        print(f"[Flux2Dev] ✅ Cleanup complete")
    
    def unload_from_gpu(self):
        """
        停止所有工作进程（清理资源）
        """
        if not hasattr(self, 'processes') or len(self.processes) == 0:
            print(f"[Flux2Dev] No processes to unload")
            return
        
        print(f"[Flux2Dev] Stopping {len(self.processes)} worker processes...")
        
        # 停止结果分发线程（如果正在运行）
        if hasattr(self, '_result_dispatcher_thread'):
            self._stop_result_dispatcher()
        
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
                print(f"[Flux2Dev] ✓ GPU {gpu_id} process already stopped")
                continue
            
            try:
                remaining_time = max(1, 5 - (time.time() - start_time))
                p.join(timeout=remaining_time)
                
                if p.is_alive():
                    print(f"[Flux2Dev] ⚠️  GPU {gpu_id} process did not terminate gracefully, forcing...")
                    p.terminate()
                    p.join(timeout=3)
                    
                    # 如果terminate还不行，使用kill
                    if p.is_alive():
                        print(f"[Flux2Dev] ⚠️  GPU {gpu_id} process did not respond to SIGTERM, killing...")
                        p.kill()
                        p.join(timeout=2)
                        
                        if p.is_alive():
                            print(f"[Flux2Dev] ❌ GPU {gpu_id} process is unresponsive (zombie)")
                        else:
                            print(f"[Flux2Dev] ✅ GPU {gpu_id} process killed")
                    else:
                        print(f"[Flux2Dev] ✅ GPU {gpu_id} process terminated")
                else:
                    print(f"[Flux2Dev] ✅ GPU {gpu_id} process stopped gracefully")
                    
            except Exception as e:
                print(f"[Flux2Dev] ⚠️  Error stopping GPU {gpu_id} process: {e}")
        
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
                    print(f"[Flux2Dev] 🧹 Cleared {cleared} pending tasks from GPU {self.device_ids[i]} queue")
        
        if hasattr(self, 'result_queue'):
            cleared = 0
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                    cleared += 1
                except:
                    break
            if cleared > 0:
                print(f"[Flux2Dev] 🧹 Cleared {cleared} pending results from result queue")
        
        print(f"[Flux2Dev] ✅ All worker processes stopped")
    
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
                print(f"[Flux2Dev] ✅ All {len(self.processes)} worker processes are already running")
                return
            else:
                print(f"[Flux2Dev] ⚠️  Detected {len(dead_processes)} dead processes: {dead_processes}")
                print(f"[Flux2Dev] 🔄 Restarting all worker processes...")
                # 清理所有进程和队列
                self._cleanup_processes()
        
        # 重新初始化（启动新进程）
        print(f"[Flux2Dev] 🚀 Initializing worker processes...")
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
