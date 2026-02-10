"""
Ovis-U1 图像编辑模型实现（支持多GPU并行 - Subprocess模式）

基于 AIDC-AI/Ovis-U1-3B
使用subprocess架构实现多GPU并行（需要配置conda环境）

架构特点：
- 使用 AutoModelForCausalLM + trust_remote_code=True 加载
- 三条件引导：no_both_cond（空白）、no_txt_cond（仅图像）、cond（图像+文本）
- 自定义预处理：smart_resize、process_image_aspectratio、preprocess_inputs
- 通过临时文件传递数据，支持在特定conda环境中运行
"""

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
from concurrent.futures import ThreadPoolExecutor

from ..base_diffusion import BaseDiffusionModel


def _image_to_base64(image: Image.Image) -> str:
    """将PIL Image转换为base64字符串"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _base64_to_image(b64_str: str) -> Image.Image:
    """将base64字符串转换为PIL Image"""
    image_data = base64.b64decode(b64_str)
    return Image.open(BytesIO(image_data))


class OvisU1Model(BaseDiffusionModel):
    """
    Ovis-U1 图像编辑模型（多GPU并行 - Subprocess模式）
    
    使用subprocess实现数据并行：
    - 每个GPU启动独立的subprocess
    - 每个subprocess加载完整的模型副本
    - 任务按轮询方式分配到各个GPU
    - 所有GPU并行处理不同的图像
    
    特点：
    - 仅支持subprocess模式（需要配置conda_env）
    - 支持多GPU并行处理
    - 使用三条件引导机制
    - 支持 img_cfg 和 txt_cfg 参数
    """
    
    def _initialize(self):
        """初始化模型配置"""
        # 获取模型路径配置
        self.model_path = self.config.get("model_path", "AIDC-AI/Ovis-U1-3B")
        self.ovis_repo = self.config.get("ovis_repo", "/data2/yixuan/Ovis-U1")
        
        # GPU配置
        device_ids = self.config.get("device_ids", None)
        
        # 检测conda环境配置
        self.conda_env = self.config.get("conda_env", None)
        if self.conda_env is None:
            raise ValueError("Ovis-U1 requires 'conda_env' parameter (subprocess mode only)")
        
        # 确定使用哪些GPU
        if device_ids is None:
            self.num_gpus = torch.cuda.device_count()
            self.device_ids = list(range(self.num_gpus))
        else:
            self.device_ids = device_ids
            self.num_gpus = len(device_ids)
        
        print(f"[Ovis-U1] 检测到 {torch.cuda.device_count()} 个GPU")
        print(f"[Ovis-U1] 将使用 {self.num_gpus} 个GPU: {self.device_ids}")
        
        # 初始化subprocess模式
        self._initialize_subprocess_mode()
    
    def _initialize_subprocess_mode(self):
        """初始化subprocess模式"""
        print(f"[Ovis-U1] 使用Subprocess模式（conda环境: {self.conda_env}）\n")
        
        # 查找worker脚本路径
        current_dir = Path(__file__).parent.parent
        self.worker_script = current_dir / "ovis_u1_subprocess_worker.py"
        
        if not self.worker_script.exists():
            raise FileNotFoundError(f"Worker script not found: {self.worker_script}")
        
        # 验证Ovis仓库路径（可选）
        ovis_repo = Path(self.ovis_repo)
        if ovis_repo.exists():
            print(f"[Ovis-U1] Ovis repo found: {self.ovis_repo}")
        else:
            print(f"[Ovis-U1] Warning: Ovis repo not found at {self.ovis_repo}, will use HuggingFace")
        
        print("=" * 70)
        print("🚀 Ovis-U1 Subprocess Mode Initialized")
        print("=" * 70)
        print(f"  Conda Environment: {self.conda_env}")
        print(f"  Worker Script: {self.worker_script}")
        print(f"  Model Path: {self.model_path}")
        print(f"  Ovis Repo: {self.ovis_repo}")
        print(f"  GPUs: {self.device_ids}")
        print("  ⚡ Models will be loaded on-demand in subprocess")
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
        results = self.batch_edit([original_image], [edit_instruction], **kwargs)
        return results[0]
    
    def batch_edit(self, images: List[Image.Image],
                   instructions: List[str],
                   **kwargs) -> List[Image.Image]:
        """
        多GPU并行批量编辑图像
        
        Args:
            images: 原始图像列表
            instructions: 编辑指令列表
            **kwargs: 其他参数
            
        Returns:
            编辑后的图像列表
        """
        if len(images) != len(instructions):
            raise ValueError("Number of images must match number of instructions")
        
        return self._batch_edit_subprocess(images, instructions, **kwargs)
    
    def _batch_edit_subprocess(self, images: List[Image.Image],
                                instructions: List[str],
                                **kwargs) -> List[Image.Image]:
        """
        Subprocess模式批量编辑
        
        将任务分配到多个GPU，每个GPU启动一个subprocess执行
        """
        n = len(images)
        num_gpus = self.num_gpus
        base_seed = kwargs.get("seed", self.config.get("seed", 42))
        
        print(f"\n[Ovis-U1-Subprocess] Starting batch edit: {n} images on {num_gpus} GPUs")
        print(f"  🐍 Conda Environment: {self.conda_env}")
        
        # 编码所有图像
        print(f"[Ovis-U1-Subprocess] Encoding {n} images to base64...")
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
                '--model-path', self.model_path,
                '--ovis-repo', self.ovis_repo,
                '--device', f'cuda:{gpu_id}',
                '--dtype', self.config.get('dtype', 'bfloat16'),
                '--num-inference-steps', str(self.config.get('num_inference_steps', 50)),
                '--img-cfg', str(self.config.get('img_cfg', 1.5)),
                '--txt-cfg', str(self.config.get('txt_cfg', 6.0)),
                '--seed', str(self.config.get('seed', 42)),
            ])
            
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
            return_code = process.wait(timeout=7200)  # 2小时超时
            
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
    
    def load_to_gpu(self, parallel: bool = True):
        """
        Subprocess模式不需要预加载模型
        """
        print(f"[Ovis-U1] Subprocess mode: models loaded on-demand")
    
    def unload_from_gpu(self):
        """
        Subprocess模式：资源在subprocess结束时自动释放
        """
        print(f"[Ovis-U1] Subprocess mode: resources auto-released")














