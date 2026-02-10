"""
Step1X-Edit diffusion model implementation
Step1X-Edit 图像编辑模型实现（支持多GPU并行 - 多进程版本）

基于 stepfun-ai/Step1X-Edit-v1p2
使用多进程架构实现多GPU并行

多进程架构：
- 每个GPU对应一个独立进程
- 进程间完全隔离，避免GIL和资源竞争
- 使用Queue进行进程间通信

Subprocess模式（当配置conda_env时启用）：
- 使用subprocess调用独立脚本
- 支持在特定conda环境中运行
- 通过临时文件传递数据
"""

import multiprocessing as mp
import subprocess
import tempfile
import json
import torch
from PIL import Image
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import base64
from io import BytesIO
import sys
from concurrent.futures import ThreadPoolExecutor

from ..base_diffusion import BaseDiffusionModel
from ....utils import setup_logger

# 必须设置，否则多进程会出错（仅在非subprocess模式下需要）
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过


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
    print(f"[GPU {gpu_id}] 🔄 Loading Step1X-Edit model...")
    try:
        from diffusers import Step1XEditPipelineV1P2
        
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
        print(f"[GPU {gpu_id}] 🔹 Loading Step1X-Edit pipeline...")
        pipeline = Step1XEditPipelineV1P2.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        
        # 移动到目标GPU
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
        true_cfg_scale = config.get("true_cfg_scale", 6.0)
        seed = config.get("seed", 0)
        enable_thinking_mode = config.get("enable_thinking_mode", False)
        enable_reflection_mode = config.get("enable_reflection_mode", False)
        
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
                use_seed = current_seed if current_seed is not None else seed
                show_progress = kwargs.get("show_progress", True)  # 默认显示进度条
                thinking_mode = kwargs.get("enable_thinking_mode", enable_thinking_mode)
                reflection_mode = kwargs.get("enable_reflection_mode", enable_reflection_mode)
                
                # 准备pipeline输入（根据官方API）
                generator = torch.Generator(device=f"cuda:{gpu_id}").manual_seed(use_seed)
                
                # Step1X-Edit的pipeline输入（根据官方API，不使用callback）
                pipeline_inputs = {
                    "image": image,
                    "prompt": instruction,
                    "num_inference_steps": num_steps,
                    "true_cfg_scale": cfg_scale,
                    "generator": generator,
                    "enable_thinking_mode": thinking_mode,
                    "enable_reflection_mode": reflection_mode,
                }
                
                # 执行推理
                if show_progress:
                    print(f"[GPU {gpu_id}] Task {task_id}: Processing {num_steps} steps...")
                
                with torch.inference_mode():
                    pipe_output = pipeline(**pipeline_inputs)
                    # Step1X-Edit返回的是pipe_output.final_images[0]
                    edited_image = pipe_output.images[0]
                
                if show_progress:
                    print(f"[GPU {gpu_id}] Task {task_id}: Completed ✓")
                
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
    


class Step1XEditModel(BaseDiffusionModel):
    """
    Step1X-Edit 图像编辑模型（多GPU并行 - 多进程版本）
    
    使用multiprocessing实现数据并行：
    - 每个GPU对应一个独立进程
    - 每个进程加载一个完整的模型副本
    - 任务按轮询方式分配到各个GPU进程
    - 所有GPU进程并行处理不同的图像
    
    特点：
    - 支持多GPU并行处理
    - 支持批次同步，确保GPU间进度一致
    - 进程间完全隔离，避免GIL和资源竞争
    - 支持subprocess模式（使用特定conda环境）
    """
    
    def _initialize(self):
        """初始化多GPU模型"""
        # 获取配置
        self.model_name = self.config.get("model_name", "stepfun-ai/Step1X-Edit-v1p2")
        device_ids = self.config.get("device_ids", None)
        
        # 检测是否使用subprocess模式（当配置了conda_env时启用）
        self.conda_env = self.config.get("conda_env", None)
        self.use_subprocess = self.conda_env is not None
        
        # 确定使用哪些GPU
        if device_ids is None:
            self.num_gpus = torch.cuda.device_count()
            self.device_ids = list(range(self.num_gpus))
        else:
            self.device_ids = device_ids
            self.num_gpus = len(device_ids)
        
        print(f"[Step1XEdit] 检测到 {torch.cuda.device_count()} 个GPU")
        print(f"[Step1XEdit] 将使用 {self.num_gpus} 个GPU: {self.device_ids}")
        
        if self.use_subprocess:
            # Subprocess模式：使用特定conda环境
            self._initialize_subprocess_mode()
        else:
            # 原有的multiprocessing模式
            self._initialize_multiprocessing_mode()
    
    def _initialize_subprocess_mode(self):
        """初始化subprocess模式（使用特定conda环境）"""
        print(f"[Step1XEdit] 使用Subprocess模式（conda环境: {self.conda_env}）\n")
        
        # 查找worker脚本路径
        current_dir = Path(__file__).parent.parent
        self.worker_script = current_dir / "step1x_edit_subprocess_worker.py"
        
        if not self.worker_script.exists():
            raise FileNotFoundError(f"Worker script not found: {self.worker_script}")
        
        # subprocess模式不需要预启动进程，每次batch_edit时动态调用
        self.processes = []  # 保持兼容性
        self.task_queues = []
        self.result_queue = None
        
        print("=" * 70)
        print("🚀 Step1X-Edit Subprocess Mode Initialized")
        print("=" * 70)
        print(f"  Conda Environment: {self.conda_env}")
        print(f"  Worker Script: {self.worker_script}")
        print(f"  GPUs: {self.device_ids}")
        print("  ⚡ Models will be loaded on-demand in subprocess")
        print("=" * 70)
        print()
    
    def _initialize_multiprocessing_mode(self):
        """初始化原有的multiprocessing模式"""
        print(f"[Step1XEdit] 使用多进程架构（每个GPU一个独立进程）\n")
        
        # 创建进程间通信队列
        self.task_queues = [mp.Queue() for _ in range(self.num_gpus)]
        self.result_queue = mp.Queue()
        
        # 启动工作进程
        self.processes = []
        print("=" * 70)
        print("🚀 Starting Worker Processes (Step1X-Edit)")
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
    
    def edit_image(self, original_image: Image.Image, 
                   edit_instruction: str,
                   **kwargs) -> Image.Image:
        """
        编辑单张图像（使用第一个GPU进程）
        
        Args:
            original_image: 原始PIL图像
            edit_instruction: 编辑指令
            **kwargs: 其他参数
            
        Returns:
            编辑后的PIL图像
        """
        # 单张图像使用第一个GPU进程
        results = self.batch_edit([original_image], [edit_instruction], **kwargs)
        return results[0]
    
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
            
        Returns:
            编辑后的图像列表
        """
        if len(images) != len(instructions):
            raise ValueError("Number of images must match number of instructions")
        
        # 根据模式选择实现
        if self.use_subprocess:
            return self._batch_edit_subprocess(images, instructions, **kwargs)
        else:
            return self._batch_edit_multiprocessing(images, instructions, **kwargs)
    
    def _batch_edit_multiprocessing(self, images: List[Image.Image],
                                     instructions: List[str],
                                     **kwargs) -> List[Image.Image]:
        """原有的multiprocessing模式批量编辑"""
        n = len(images)
        num_gpus = self.num_gpus
        enable_sync = kwargs.pop("enable_batch_sync", True)
        
        print(f"\n[Step1XEdit] Starting batch edit: {n} images on {num_gpus} GPUs")
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
        
        # 获取基础seed
        base_seed = kwargs.get("seed", self.config.get("seed", 0))
        
        if enable_sync:
            results = self._batch_edit_with_sync(
                images, instructions, n, num_gpus, base_seed, **kwargs
            )
        else:
            results = self._batch_edit_no_sync(
                images, instructions, n, num_gpus, base_seed, **kwargs
            )
        
        print(f"✅ Batch edit completed: {n} images\n")
        return results
    
    def _batch_edit_subprocess(self, images: List[Image.Image],
                                instructions: List[str],
                                **kwargs) -> List[Image.Image]:
        """
        Subprocess模式批量编辑（使用特定conda环境）
        
        将任务分配到多个GPU，每个GPU启动一个subprocess执行
        """
        n = len(images)
        num_gpus = self.num_gpus
        base_seed = kwargs.get("seed", self.config.get("seed", 0))
        
        print(f"\n[Step1XEdit-Subprocess] Starting batch edit: {n} images on {num_gpus} GPUs")
        print(f"  🐍 Conda Environment: {self.conda_env}")
        
        # 编码所有图像
        print(f"[Step1XEdit-Subprocess] Encoding {n} images to base64...")
        image_b64s = [_image_to_base64(img) for img in images]
        
        # 按GPU分配任务
        gpu_tasks = [[] for _ in range(num_gpus)]
        for i in range(n):
            gpu_idx = i % num_gpus
            gpu_tasks[gpu_idx].append({
                'task_id': i,
                'image_b64': image_b64s[i],
                'instruction': instructions[i],
                'seed': base_seed + i
            })
        
        # 显示任务分配
        print("=" * 70)
        print("📋 Task Assignment (Subprocess Mode):")
        print("=" * 70)
        for gpu_idx, gpu_id in enumerate(self.device_ids):
            num_tasks = len(gpu_tasks[gpu_idx])
            print(f"  GPU {gpu_id}: {num_tasks} tasks")
        print("=" * 70)
        print()
        
        # 结果列表
        results = [None] * n
        
        # 并行启动subprocess
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_idx, gpu_id in enumerate(self.device_ids):
                if gpu_tasks[gpu_idx]:
                    future = executor.submit(
                        self._call_subprocess_single_gpu,
                        gpu_tasks[gpu_idx],
                        gpu_id
                    )
                    futures.append((future, gpu_idx, gpu_id))
            
            # 收集结果
            for future, gpu_idx, gpu_id in futures:
                try:
                    gpu_results = future.result()
                    for result in gpu_results:
                        task_id = result['task_id']
                        if result['success']:
                            results[task_id] = _base64_to_image(result['image_b64'])
                        else:
                            print(f"❌ Task {task_id} failed on GPU {gpu_id}: {result['error']}")
                            results[task_id] = images[task_id]  # fallback to original
                except Exception as e:
                    print(f"❌ Error in GPU {gpu_id} subprocess: {e}")
                    # 对该GPU的所有任务使用原图作为fallback
                    for task in gpu_tasks[gpu_idx]:
                        task_id = task['task_id']
                        if results[task_id] is None:
                            results[task_id] = images[task_id]
        
        print(f"✅ Batch edit (subprocess) completed: {n} images\n")
        return results
    
    def _call_subprocess_single_gpu(self, tasks: List[Dict], gpu_id: int) -> List[Dict]:
        """
        在指定GPU上调用subprocess执行编辑任务
        
        Args:
            tasks: 任务列表，每个任务包含task_id, image_b64, instruction, seed
            gpu_id: GPU ID
            
        Returns:
            结果列表，每个结果包含task_id, success, image_b64, error
        """
        if not tasks:
            return []
        
        # 创建临时文件
        input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        try:
            # 写入输入数据
            input_data = {'tasks': tasks}
            json.dump(input_data, input_file)
            input_file.close()
            output_file.close()
            
            # 构建命令
            cmd = [
                'conda', 'run', '-n', self.conda_env, '--no-capture-output',
                'python', str(self.worker_script)
            ]
            
            # 添加参数
            cmd.extend([
                '--input', input_file.name,
                '--output', output_file.name,
                '--model-name', self.model_name,
                '--device', f'cuda:{gpu_id}',
                '--dtype', self.config.get('dtype', 'bfloat16'),
                '--num-inference-steps', str(self.config.get('num_inference_steps', 50)),
                '--true-cfg-scale', str(self.config.get('true_cfg_scale', 6.0)),
                '--seed', str(self.config.get('seed', 0)),
            ])
            
            if self.config.get('enable_thinking_mode', False):
                cmd.append('--enable-thinking-mode')
            if self.config.get('enable_reflection_mode', False):
                cmd.append('--enable-reflection-mode')
            if self.config.get('disable_progress_bar', True):
                cmd.append('--disable-progress-bar')
            
            print(f"[GPU {gpu_id}] Starting subprocess with {len(tasks)} tasks...")
            
            # 执行subprocess（实时输出）
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 实时打印stderr
            while True:
                stderr_line = process.stderr.readline()
                if stderr_line:
                    print(f"[GPU {gpu_id}] {stderr_line.rstrip()}")
                elif process.poll() is not None:
                    break
            
            # 获取剩余输出
            remaining_stderr = process.stderr.read()
            if remaining_stderr:
                for line in remaining_stderr.split('\n'):
                    if line.strip():
                        print(f"[GPU {gpu_id}] {line}")
            
            # 等待完成
            return_code = process.wait(timeout=3600)  # 1小时超时
            
            if return_code != 0:
                raise RuntimeError(f"Subprocess failed with return code {return_code}")
            
            # 读取输出
            with open(output_file.name, 'r') as f:
                output_data = json.load(f)
            
            if output_data.get('status') != 'success':
                raise RuntimeError(f"Worker error: {output_data.get('error', 'Unknown')}")
            
            return output_data['results']
            
        finally:
            # 清理临时文件
            Path(input_file.name).unlink(missing_ok=True)
            Path(output_file.name).unlink(missing_ok=True)
    
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
        
        print(f"[Step1XEdit] 🧹 Cleaning up {len(self.processes)} worker processes...")
        
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
                    print(f"[Step1XEdit]   ✅ GPU {gpu_id} process terminated")
                except Exception as e:
                    print(f"[Step1XEdit]   ⚠️  Error terminating GPU {gpu_id} process: {e}")
            else:
                print(f"[Step1XEdit]   ✓ GPU {gpu_id} process already dead")
        
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
        
        print(f"[Step1XEdit] ✅ Cleanup complete")
    
    def unload_from_gpu(self):
        """
        停止所有工作进程（清理资源）
        """
        if self.use_subprocess:
            # Subprocess模式：无需清理，模型在subprocess结束时自动释放
            print(f"[Step1XEdit] Subprocess mode: resources auto-released")
            return
        
        if not hasattr(self, 'processes') or len(self.processes) == 0:
            print(f"[Step1XEdit] No processes to unload")
            return
        
        # Multiprocessing模式：停止工作进程
        print(f"[Step1XEdit] Stopping {len(self.processes)} worker processes...")
        
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
                print(f"[Step1XEdit] ✓ GPU {gpu_id} process already stopped")
                continue
            
            try:
                remaining_time = max(1, 5 - (time.time() - start_time))
                p.join(timeout=remaining_time)
                
                if p.is_alive():
                    print(f"[Step1XEdit] ⚠️  GPU {gpu_id} process did not terminate gracefully, forcing...")
                    p.terminate()
                    p.join(timeout=3)
                    
                    # 如果terminate还不行，使用kill
                    if p.is_alive():
                        print(f"[Step1XEdit] ⚠️  GPU {gpu_id} process did not respond to SIGTERM, killing...")
                        p.kill()
                        p.join(timeout=2)
                        
                        if p.is_alive():
                            print(f"[Step1XEdit] ❌ GPU {gpu_id} process is unresponsive (zombie)")
                        else:
                            print(f"[Step1XEdit] ✅ GPU {gpu_id} process killed")
                    else:
                        print(f"[Step1XEdit] ✅ GPU {gpu_id} process terminated")
                else:
                    print(f"[Step1XEdit] ✅ GPU {gpu_id} process stopped gracefully")
                    
            except Exception as e:
                print(f"[Step1XEdit] ⚠️  Error stopping GPU {gpu_id} process: {e}")
        
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
                    print(f"[Step1XEdit] 🧹 Cleared {cleared} pending tasks from GPU {self.device_ids[i]} queue")
        
        if hasattr(self, 'result_queue'):
            cleared = 0
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                    cleared += 1
                except:
                    break
            if cleared > 0:
                print(f"[Step1XEdit] 🧹 Cleared {cleared} pending results from result queue")
        
        print(f"[Step1XEdit] ✅ All worker processes stopped")
    
    def load_to_gpu(self, parallel: bool = True):
        """
        将模型加载到GPU（重新启动worker进程/subprocess）
        
        多进程版本需要检查进程是否存活，如果进程已死则重新启动
        """
        if self.use_subprocess:
            # Subprocess模式不需要维护长期运行的进程，每次调用时启动
            print(f"[Step1XEdit] Subprocess mode: models loaded on-demand")
            return
        
        # 多进程模式：检查进程是否存活
        if hasattr(self, 'processes') and len(self.processes) > 0:
            alive_processes = [p for _, p in self.processes if p.is_alive()]
            dead_processes = [gpu_id for gpu_id, p in self.processes if not p.is_alive()]
            
            if len(alive_processes) == len(self.processes):
                print(f"[Step1XEdit] ✅ All {len(self.processes)} worker processes are already running")
                return
            else:
                print(f"[Step1XEdit] ⚠️  Detected {len(dead_processes)} dead processes: {dead_processes}")
                print(f"[Step1XEdit] 🔄 Restarting all worker processes...")
                # 清理所有进程和队列
                self._cleanup_processes()
        
        # 重新初始化（启动新进程）
        print(f"[Step1XEdit] 🚀 Initializing worker processes...")
        self._initialize()
    
    def __del__(self):
        """清理资源（析构函数）"""
        if hasattr(self, 'use_subprocess') and self.use_subprocess:
            # Subprocess模式：无需额外清理
            return
        
        if hasattr(self, 'processes') and len(self.processes) > 0:
            # Multiprocessing模式：确保所有进程都已停止（使用强制方式）
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
