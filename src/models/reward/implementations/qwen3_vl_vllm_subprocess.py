"""
Qwen3-VL vLLM Subprocess Reward Model
使用vLLM引擎在独立虚拟环境中运行评分

架构设计：
- 主进程：Python subprocess调用
- 子进程：conda环境 yx_vllm 中运行 vLLM
- 通信：临时JSON文件
- 并行：vLLM内部tensor并行（单一进程，多GPU）
"""

import json
import tempfile
import subprocess
import time
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from io import BytesIO

from ..base_reward import BaseRewardModel
from ....utils import setup_logger


class Qwen3VLvLLMSubprocessRewardModel(BaseRewardModel):
    """
    vLLM版本的Qwen3-VL Subprocess Reward Model
    
    与 Qwen3VLMultiGPUSubprocessRewardModel 的关键差异：
    1. 使用vLLM引擎（tensor并行）而非transformer（数据并行）
    2. 单一subprocess，所有GPU通过tensor并行共享
    3. 更大的batch_size（vLLM支持更高效的批量推理）
    4. 运行在 yx_vllm 虚拟环境
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = setup_logger(self.__class__.__name__)
        
        # 基本配置
        self.model_name = config.get("model_name", "Qwen/Qwen3-VL-30B-Instruct")
        self.dtype = config.get("dtype", "bfloat16")
        self.max_new_tokens = config.get("max_new_tokens", 128)
        self.batch_size = config.get("batch_size", 8)  # vLLM支持更大batch
        self.temperature = config.get("temperature", 0)
        self.script_path = config.get("script_path", None)
        self.conda_env = config.get("conda_env", "yx_vllm")  # 默认使用yx_vllm环境
        self.python_path = config.get("python_path", None)
        self.timeout = config.get("timeout", 1200)  # vLLM加载较慢，增加超时时间
        
        # Tensor并行配置
        self.tensor_parallel_size = config.get("tensor_parallel_size", None)  # None则自动检测
        
        # 日志文件路径（用于子进程输出）
        self.log_file_path = config.get("log_file_path", None)
        
        super().__init__(config)
    
    def _initialize(self):
        """初始化模型（实现BaseModel的抽象方法）"""
        # 检测脚本路径
        if self.script_path is None:
            current_dir = Path(__file__).parent.parent
            self.script_path = current_dir / "qwen3_vl_vllm_standalone.py"
        else:
            self.script_path = Path(self.script_path)
        
        if not self.script_path.exists():
            raise FileNotFoundError(f"Standalone script not found: {self.script_path}")
        
        self.logger.info(f"Initialized Qwen3-VL vLLM Subprocess Reward Model")
        self.logger.info(f"  Model: {self.model_name}")
        self.logger.info(f"  Tensor Parallel Size: {self.tensor_parallel_size if self.tensor_parallel_size else 'auto-detect'}")
        self.logger.info(f"  Batch Size: {self.batch_size}")
        self.logger.info(f"  Script: {self.script_path}")
        self.logger.info(f"  Conda Env: {self.conda_env}")
    
    def _image_to_data_uri(self, pil_image: Image.Image) -> str:
        """
        将PIL图像转换为data URI格式（vLLM要求）
        
        Args:
            pil_image: PIL图像对象
            
        Returns:
            data URI字符串
        """
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        b64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64_string}"
    
    def _call_subprocess(self, tasks: List[Dict]) -> List[Dict]:
        """
        调用子进程进行评分
        
        Args:
            tasks: 评分任务列表
            
        Returns:
            评分结果列表
        """
        if not tasks:
            return []
        
        # 创建临时文件
        input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
        
        try:
            # 写入输入数据
            input_data = {'tasks': tasks}
            json.dump(input_data, input_file, ensure_ascii=False)
            input_file.close()
            output_file.close()
            
            # 构建命令
            if self.conda_env:
                cmd = [
                    'conda', 'run', '-n', self.conda_env, '--no-capture-output',
                    'python', str(self.script_path)
                ]
            elif self.python_path:
                cmd = [self.python_path, str(self.script_path)]
            else:
                cmd = ['python', str(self.script_path)]
            
            # 添加参数
            cmd.extend([
                '--input', input_file.name,
                '--output', output_file.name,
                '--model-name', self.model_name,
                '--batch-size', str(self.batch_size),
                '--max-new-tokens', str(self.max_new_tokens),
                '--temperature', str(self.temperature),
                '--dtype', self.dtype,
            ])
            
            # 添加tensor_parallel_size（如果指定）
            if self.tensor_parallel_size is not None:
                cmd.extend(['--tensor-parallel-size', str(self.tensor_parallel_size)])
            
            self.logger.info(f"Starting vLLM subprocess with {len(tasks)} tasks...")
            self.logger.info(f"Command: {' '.join(cmd[:5])} ... (truncated)")
            
            # 执行子进程（使用Popen实时捕获输出）
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 实时打印stderr（包含评分进度），同时写入日志文件
            stderr_lines = []
            
            def read_stderr():
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    line = line.rstrip()
                    stderr_lines.append(line)
                    # 输出到日志（添加[vLLM]前缀）
                    self.logger.info(f"[vLLM] {line}")
                    # 如果有log_file_path，也写入文件
                    if self.log_file_path:
                        try:
                            with open(self.log_file_path, 'a', encoding='utf-8') as log_f:
                                log_f.write(f"[vLLM] {line}\n")
                        except:
                            pass
            
            import threading
            stderr_thread = threading.Thread(target=read_stderr)
            stderr_thread.start()
            
            # 等待进程结束
            try:
                return_code = process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stderr_thread.join(timeout=5)
                raise RuntimeError(f"vLLM subprocess timed out after {self.timeout}s")
            
            stderr_thread.join()
            
            if return_code != 0:
                error_msg = '\n'.join(stderr_lines[-10:]) if stderr_lines else "Unknown error"
                raise RuntimeError(f"vLLM subprocess failed with code {return_code}: {error_msg}")
            
            # 读取输出文件
            with open(output_file.name, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            
            # 检查状态
            if output_data.get('status') != 'success':
                raise RuntimeError(f"vLLM error: {output_data.get('error', 'Unknown')}")
            
            return output_data['scores']
        
        finally:
            # 清理临时文件
            Path(input_file.name).unlink(missing_ok=True)
            Path(output_file.name).unlink(missing_ok=True)
    
    def score(self, *args, **kwargs) -> str:
        """评分单张图像（不建议使用，建议使用batch_score）"""
        results = self.batch_score([args], [kwargs])
        return results[0]['score'] if results else "no"
    
    def batch_score(self,
                   edited_images: List[Image.Image],
                   original_descriptions: List[str],
                   edit_instructions: List[str],
                   system_prompts: List[str],
                   user_prompts: List[str],
                   original_images: Optional[List[Image.Image]] = None,
                   rationales: Optional[List[str]] = None,
                   return_failures: bool = False,
                   **kwargs):
        """
        批量评分（vLLM引擎）
        
        Args:
            edited_images: 编辑后的图像列表
            original_descriptions: 原始图像描述列表（占位，vLLM不直接使用）
            edit_instructions: 编辑指令列表（占位，已在prompts中）
            system_prompts: 系统提示列表
            user_prompts: 用户提示列表
            original_images: 原始图像列表（可选）
            rationales: 编辑任务难点描述列表（可选）
            return_failures: 是否返回失败信息（默认False保持向后兼容）
            
        Returns:
            如果return_failures=False: Tuple[List[str], List[str]]
            如果return_failures=True: Tuple[List[str], List[str], Dict]
        """
        n = len(edited_images)
        self.logger.info(f"vLLM batch scoring {n} images (tensor_parallel across GPUs)...")
        
        # 编码所有图像为data URI
        self.logger.info(f"Encoding images to data URI format...")
        start_time = time.time()
        
        edited_image_uris = [self._image_to_data_uri(img) for img in edited_images]
        
        # 编码原图（如果提供）
        original_image_uris = None
        if original_images is not None and len(original_images) > 0:
            if len(original_images) != n:
                self.logger.warning(
                    f"Original images count ({len(original_images)}) doesn't match edited images count ({n}). "
                    f"Will only use first {min(len(original_images), n)} original images."
                )
            original_image_uris = [self._image_to_data_uri(img) for img in original_images[:n]]
            self.logger.info(f"Original images provided: {len(original_image_uris)} images encoded")
        else:
            self.logger.info("No original images provided")
        
        encode_time = time.time() - start_time
        self.logger.info(f"Encoding completed in {encode_time:.2f}s")
        
        # 准备rationales（如果未提供）
        if rationales is None:
            rationales = [None] * n
        
        # 准备所有任务
        all_tasks = []
        for i in range(n):
            task = {
                'edited_image_b64': edited_image_uris[i],
                'system_prompt': system_prompts[i],
                'user_prompt': user_prompts[i],
                'rationale': rationales[i],
            }
            # 如果提供了原图，添加到task中
            if original_image_uris is not None and i < len(original_image_uris):
                task['original_image_b64'] = original_image_uris[i]
            all_tasks.append(task)
        
        self.logger.info(f"Prepared {len(all_tasks)} tasks for vLLM processing")
        
        # 调用subprocess
        try:
            results = self._call_subprocess(all_tasks)
            
            # 解析结果
            scores = []
            reasonings = []
            failures = {
                'failed_gpus': [],  # vLLM不需要GPU级别追踪，保持接口兼容
                'failed_sample_indices': [],
                'failed_sample_count': 0,
                'total_samples': n,
                'error_messages': {}
            }
            
            for i, result in enumerate(results):
                score = result.get('score')
                reasoning = result.get('reasoning', '')
                
                # 检查是否失败
                if score is None or result.get('error'):
                    scores.append('no')  # 默认值
                    reasonings.append('')
                    failures['failed_sample_indices'].append(i)
                    failures['failed_sample_count'] += 1
                    if result.get('error'):
                        failures['error_messages'][f'sample_{i}'] = result['error']
                else:
                    scores.append(score)
                    reasonings.append(reasoning)
            
            self.logger.info(f"vLLM scoring completed!")
            
            # 如果有失败，输出警告
            if failures['failed_sample_count'] > 0:
                self.logger.warning(
                    f"⚠️  {failures['failed_sample_count']}/{n} samples failed, using default result 'no'"
                )
            
            if return_failures:
                return scores, reasonings, failures
            else:
                return scores, reasonings
        
        except Exception as e:
            self.logger.error(f"vLLM subprocess failed: {e}")
            
            # 全部失败时返回默认值
            scores = ['no'] * n
            reasonings = [''] * n
            failures = {
                'failed_gpus': [],
                'failed_sample_indices': list(range(n)),
                'failed_sample_count': n,
                'total_samples': n,
                'error_messages': {'subprocess': str(e)}
            }
            
            if return_failures:
                return scores, reasonings, failures
            else:
                return scores, reasonings
