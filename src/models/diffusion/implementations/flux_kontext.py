"""
Flux.1-Kontext diffusion model implementation
Flux.1-Kontext 图像编辑模型实现（支持多GPU并行）

基于 black-forest-labs/FLUX.1-Kontext-dev
"""

import threading
import torch
from PIL import Image
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..base_diffusion import BaseDiffusionModel
from ....utils import setup_logger


# 全局锁，用于序列化模型加载过程（避免OOM）
_model_load_lock = threading.Lock()


class FluxKontextGPUWorker:
    """Flux Kontext GPU工作器类，每个实例绑定到一个GPU"""
    
    def __init__(self, gpu_id: int, model_name: str, config: Dict[str, Any]):
        """
        初始化GPU Worker
        
        Args:
            gpu_id: GPU ID
            model_name: 模型名称或路径
            config: 模型配置参数
        """
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.model_name = model_name
        self.config = config
        self.pipeline = None
        self._model_loaded = False
        
        # 提取配置参数
        self.dtype = config.get("dtype", "bfloat16")
        self.num_inference_steps = config.get("num_inference_steps", 28)  # Flux默认28步
        self.guidance_scale = config.get("guidance_scale", 2.5)
        self.seed = config.get("seed", 0)
        self.disable_progress_bar = config.get("disable_progress_bar", True)
        self.height = config.get("height", None)  # 可选：输出高度
        self.width = config.get("width", None)   # 可选：输出宽度
    
    def _load_model_serial(self):
        """
        串行加载模型：使用全局锁确保一次只有一个GPU在加载
        
        这样可以避免多个GPU同时加载导致OOM
        """
        if self._model_loaded:
            return True
        
        # 使用全局锁，确保一次只有一个GPU在加载模型
        with _model_load_lock:
            # 双重检查
            if self._model_loaded:
                return True
            
            print(f"[GPU {self.gpu_id}] 🔄 Loading Flux.1-Kontext model...")
            try:
                from diffusers import FluxKontextPipeline
                
                # 设置当前设备
                torch.cuda.set_device(self.gpu_id)
                
                # 清空GPU缓存
                print(f"[GPU {self.gpu_id}] 🧹 Clearing GPU cache...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # 加载模型 - 使用low_cpu_mem_usage减少内存占用
                print(f"[GPU {self.gpu_id}] 🔹 Loading model to {self.device}...")
                self.pipeline = FluxKontextPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=getattr(torch, self.dtype),
                )
                
                # 移动到目标GPU
                self.pipeline.to(self.device)
                
                # 禁用进度条
                if self.disable_progress_bar:
                    self.pipeline.set_progress_bar_config(disable=True)
                
                self._model_loaded = True
                print(f"[GPU {self.gpu_id}] ✅ Model loaded successfully")
                return True
                
            except Exception as e:
                print(f"[GPU {self.gpu_id}] ❌ Error loading model: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def _ensure_model_loaded(self):
        """确保模型已加载"""
        if self._model_loaded:
            return True
        
        print(f"[GPU {self.gpu_id}] ⚠️  Model not pre-loaded, loading now...")
        return self._load_model_serial()
    
    def edit_image(self, original_image: Image.Image, 
                   edit_instruction: str, 
                   seed: int = None,
                   show_progress: bool = True,
                   **kwargs) -> Image.Image:
        """
        编辑单张图像
        
        Args:
            original_image: 原始PIL图像
            edit_instruction: 编辑指令
            seed: 随机种子（可选）
            show_progress: 是否显示去噪进度条
            **kwargs: 其他参数
            
        Returns:
            编辑后的PIL图像
        """
        # 确保模型已加载
        if not self._ensure_model_loaded():
            raise RuntimeError(f"[GPU {self.gpu_id}] Failed to load model")
        
        # 确保使用正确的设备
        torch.cuda.set_device(self.gpu_id)
        
        # 确保图像是RGB格式
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # 获取参数
        num_steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        guidance = kwargs.get("guidance_scale", self.guidance_scale)
        use_seed = seed if seed is not None else self.seed
        height = kwargs.get("height", self.height)
        width = kwargs.get("width", self.width)
        
        # 准备输入
        inputs = {
            "image": original_image,
            "prompt": edit_instruction,
            "generator": torch.Generator(device=self.device).manual_seed(use_seed),
            "guidance_scale": guidance,
            "num_inference_steps": num_steps,
        }
        
        # 可选：设置输出尺寸
        if height is not None:
            inputs["height"] = height
        if width is not None:
            inputs["width"] = width
        
        # 添加去噪进度条（如果启用）
        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=num_steps, desc=f"[GPU {self.gpu_id}] Denoising", 
                       unit="step", leave=False, position=self.gpu_id)
            
            def callback(pipe, step_index, timestep, callback_kwargs):
                pbar.update(1)
                return callback_kwargs
            
            inputs["callback_on_step_end"] = callback
        
        # 执行编辑
        try:
            with torch.inference_mode():
                output = self.pipeline(**inputs)
                edited_image = output.images[0]
        finally:
            if show_progress:
                pbar.close()
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        return edited_image
    
    def unload_from_gpu(self):
        """将模型从GPU卸载到CPU"""
        if self._model_loaded and self.pipeline is not None:
            print(f"[GPU {self.gpu_id}] 🔄 Unloading model from GPU...")
            self.pipeline.to('cpu')
            torch.cuda.empty_cache()
            print(f"[GPU {self.gpu_id}] ✅ Model unloaded")
    
    def load_to_gpu(self):
        """将模型从CPU加载到GPU"""
        if self._model_loaded and self.pipeline is not None:
            print(f"[GPU {self.gpu_id}] 🔄 Loading model to GPU...")
            self.pipeline.to(self.device)
            print(f"[GPU {self.gpu_id}] ✅ Model loaded to GPU")


class FluxKontextModel(BaseDiffusionModel):
    """
    Flux.1-Kontext 图像编辑模型（多GPU并行）
    
    使用ThreadPoolExecutor实现数据并行：
    - 每个GPU加载一个完整的模型副本
    - 任务按轮询方式分配到各个GPU
    - 所有GPU并行处理不同的图像
    """
    
    def _initialize(self):
        """初始化多GPU模型"""
        # 获取配置
        self.model_name = self.config.get("model_name", "black-forest-labs/FLUX.1-Kontext-dev")
        device_ids = self.config.get("device_ids", None)
        
        # 确定使用哪些GPU
        if device_ids is None:
            # 如果没有指定，使用所有可用GPU
            self.num_gpus = torch.cuda.device_count()
            self.device_ids = list(range(self.num_gpus))
        else:
            self.device_ids = device_ids
            self.num_gpus = len(device_ids)
        
        print(f"[FluxKontext] 检测到 {torch.cuda.device_count()} 个GPU")
        print(f"[FluxKontext] 将使用 {self.num_gpus} 个GPU: {self.device_ids}")
        
        # 创建GPU工作器
        self.workers = []
        for gpu_id in self.device_ids:
            worker = FluxKontextGPUWorker(
                gpu_id=gpu_id,
                model_name=self.model_name,
                config=self.config
            )
            self.workers.append(worker)
        
        print(f"[FluxKontext] 创建了 {len(self.workers)} 个GPU workers\n")
        
        # ===== 串行加载所有GPU的模型 =====
        print("=" * 70)
        print("🚀 Sequential Model Loading Phase (Flux.1-Kontext)")
        print("=" * 70)
        print(f"Loading models to {len(self.workers)} GPUs sequentially...")
        print("(All GPUs will be loaded before any processing starts)")
        print()
        
        loaded_workers = []
        for i, worker in enumerate(self.workers):
            gpu_id = worker.gpu_id
            print(f"[{i+1}/{len(self.workers)}] Loading model to GPU {gpu_id}...")
            
            try:
                success = worker._load_model_serial()
                if success:
                    loaded_workers.append(worker)
                    print(f"  ✅ GPU {gpu_id}: Model loaded and ready\n")
                else:
                    print(f"  ❌ GPU {gpu_id}: Failed to load model\n")
            except Exception as e:
                print(f"  ❌ GPU {gpu_id}: Error - {str(e)[:100]}\n")
        
        if not loaded_workers:
            raise RuntimeError("❌ ERROR: No GPUs available! Failed to load model on any GPU.")
        
        self.workers = loaded_workers
        print(f"✅ Successfully loaded models on {len(self.workers)} GPUs")
        print(f"  ⚡ All {len(self.workers)} GPUs are now ready to start processing")
        print("=" * 70)
        print()
    
    def edit_image(self, original_image: Image.Image, 
                   edit_instruction: str,
                   **kwargs) -> Image.Image:
        """
        编辑单张图像（使用第一个GPU）
        
        Args:
            original_image: 原始PIL图像
            edit_instruction: 编辑指令
            **kwargs: 其他参数
            
        Returns:
            编辑后的PIL图像
        """
        # 单张图像使用第一个GPU
        return self.workers[0].edit_image(original_image, edit_instruction, **kwargs)
    
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
        
        n = len(images)
        num_gpus = len(self.workers)
        enable_sync = kwargs.pop("enable_batch_sync", True)  # 默认启用批次同步
        
        print(f"\n[FluxKontext] Starting batch edit: {n} images on {num_gpus} GPUs")
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
        base_seed = kwargs.get("seed", self.workers[0].seed)
        
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
        批次同步模式：确保每批所有GPU完成后再开始下一批
        
        这样可以避免GPU之间进度差异累积，防止卡间通信混乱
        """
        results = [None] * n
        
        # 计算批次数
        num_batches = (n + num_gpus - 1) // num_gpus
        
        print(f"🔄 Batch synchronization mode:")
        print(f"   - Total batches: {num_batches}")
        print(f"   - Batch size: {num_gpus} (one task per GPU)")
        print(f"   - All GPUs will stay synchronized at batch boundaries\n")
        
        # 使用ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            # 总进度条
            with tqdm(total=n, desc="[SYNC] Editing images", unit="img") as pbar:
                # 逐批处理
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * num_gpus
                    batch_end = min(batch_start + num_gpus, n)
                    batch_size = batch_end - batch_start
                    
                    # 提交当前批次的任务
                    futures = []
                    indices = []
                    
                    for i in range(batch_start, batch_end):
                        worker = self.workers[(i - batch_start) % num_gpus]
                        current_seed = base_seed + i
                        
                        future = executor.submit(
                            worker.edit_image,
                            images[i],
                            instructions[i],
                            current_seed,
                            show_progress=False,  # 禁用单独的进度条
                            **kwargs
                        )
                        futures.append(future)
                        indices.append(i)
                    
                    # 等待当前批次所有任务完成（同步点）
                    for future, idx in zip(futures, indices):
                        try:
                            result = future.result()
                            results[idx] = result
                        except Exception as e:
                            print(f"\n❌ Error editing image {idx}: {e}")
                            results[idx] = images[idx]  # fallback
                        finally:
                            pbar.update(1)
                    
                    # 当前批次完成，所有GPU已同步，可以开始下一批
                    if batch_idx < num_batches - 1:
                        pbar.set_postfix_str(f"Batch {batch_idx+1}/{num_batches} done, GPUs synced ✓")
        
        return results
    
    def _batch_edit_no_sync(self, images, instructions, n, num_gpus, base_seed, **kwargs):
        """
        无同步模式：一次性提交所有任务（原始实现）
        
        适用于GPU性能一致或不关心同步的场景
        """
        results = [None] * n
        
        print(f"⚡ No-sync mode: All {n} tasks submitted at once\n")
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            # 提交所有任务
            future_to_index = {}
            
            for idx in range(n):
                worker = self.workers[idx % num_gpus]
                current_seed = base_seed + idx
                
                future = executor.submit(
                    worker.edit_image,
                    images[idx],
                    instructions[idx],
                    current_seed,
                    show_progress=False,
                    **kwargs
                )
                future_to_index[future] = idx
            
            # 收集结果（带进度条）
            with tqdm(total=n, desc="[NO-SYNC] Editing images", unit="img") as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        print(f"\n❌ Error editing image {idx}: {e}")
                        results[idx] = images[idx]
                    finally:
                        pbar.update(1)
        
        return results
    
    def unload_from_gpu(self):
        """
        将所有GPU上的模型卸载到CPU
        
        注意：卸载操作可以并行执行，因为只是释放显存，不涉及资源竞争
        """
        print(f"[FluxKontext] Unloading models from {len(self.workers)} GPUs (parallel)...")
        
        # 使用ThreadPoolExecutor并行卸载
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [executor.submit(worker.unload_from_gpu) for worker in self.workers]
            # 等待所有卸载完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[FluxKontext] ⚠️ Error during unload: {e}")
        
        print(f"[FluxKontext] All models unloaded")
    
    def load_to_gpu(self, parallel: bool = True):
        """
        将所有GPU上的模型从CPU加载回GPU
        
        Args:
            parallel: 是否并行加载。默认True（推荐）。
                     - True: 并行加载，速度快
                     - False: 串行加载，更保守但慢
        
        注意：与首次加载不同，这里是将已在内存中的模型移回GPU，
             可以安全地并行执行（不会像首次加载那样有OOM风险）
        """
        if parallel:
            print(f"[FluxKontext] Loading models to {len(self.workers)} GPUs (parallel)...")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                futures = [executor.submit(worker.load_to_gpu) for worker in self.workers]
                # 等待所有加载完成
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[FluxKontext] ⚠️ Error during load: {e}")
        else:
            print(f"[FluxKontext] Loading models to {len(self.workers)} GPUs (serial)...")
            for worker in self.workers:
                worker.load_to_gpu()
        
        print(f"[FluxKontext] All models loaded to GPU")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'workers'):
            for worker in self.workers:
                if hasattr(worker, 'pipeline') and worker.pipeline is not None:
                    del worker.pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

