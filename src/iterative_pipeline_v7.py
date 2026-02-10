"""
Iterative Refinement Pipeline
迭代优化Pipeline - 通过MLLM分析和二次编辑优化结果

期望实现v3与v4的区别：reward 打分变成了yes or no
"""

import os
import logging
import random
from typing import Dict, List, Optional, Any
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from .data.benchmark_loader import BenchmarkLoader
from .data.iterative_data import IterativeDataPair, IterativeCategoryData, IterativeBenchmarkData
from .utils.logger import setup_logger
from .utils.image_utils import decode_base64_image
from .utils.prompt_manager import PromptManager


class IterativeRefinementPipeline:
    """
    迭代优化Pipeline
    
    流程：
    1. 初次编辑（Primary Editing）
    2. MLLM分析（MLLM Analysis）
    3. 二次编辑（Refinement Editing）
    4. 对比评分（Comparative Scoring）
    5. 生成报告（Report Generation）
    """
    
    def __init__(self, config: dict):
        """
        初始化迭代优化Pipeline
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = setup_logger("iterative_pipeline", config.get("log_level", "INFO"))
        
        # 数据加载
        self.data_loader = BenchmarkLoader()
        self.prompt_manager = PromptManager(config.get("prompts", {}))
        
        # 模型实例（延迟加载，初始化时为None）
        self.primary_diffusion = None
        self.refinement_diffusion = None
        self.mllm = None
        self.reward_model = None
        
        # 当前已加载的模型（用于跟踪和自动卸载）
        self._current_loaded_model = None
        self._current_model_name = None
        
        # 保存pipeline启动时间戳（用于报告文件命名）
        from datetime import datetime
        self.start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存主模型类型（用于报告文件命名）
        self.primary_model_type = config.get("diffusion_model", {}).get("primary", {}).get("type", "unknown")
        
        # 输出配置
        self.output_dir = config["evaluation"].get("output_dir", "./results_iterative")
        self.save_images = config["evaluation"].get("save_images", True)
        self.enable_disk_cache = config["evaluation"].get("enable_disk_cache", False)
        self.cache_dir = config["evaluation"].get("cache_dir", "./cache")
        
        # 评分指标配置（控制是否进行 PQ、SC 和 Instruction Following 评价）
        self.enable_pq_metric = config["evaluation"].get("enable_pq_metric", False)  # 默认关闭
        self.enable_sc_metric = config["evaluation"].get("enable_sc_metric", False)  # 默认关闭
        self.enable_instruction_following_metric = config["evaluation"].get("enable_instruction_following_metric", False)  # 默认关闭
        
        # 是否跳过 refinement 流程（Primary Edit 后直接 Scoring，仅对 primary 评分）
        self.skip_refinement = config["evaluation"].get("skip_refinement", False)  # 默认关闭
        if self.skip_refinement:
            self.logger.info("⚠️  skip_refinement is ENABLED: Will skip Stage 2 (MLLM) and Stage 3 (Refinement)")
            self.logger.info("  Primary Edit -> Scoring -> Report. Only primary_* fields will be populated.")
        
        # Primary评分配置（控制是否对primary edited images进行评分）
        # skip_refinement 时强制 enable_primary_scoring=True
        self.enable_primary_scoring = config["evaluation"].get("enable_primary_scoring", False)  # 默认关闭以保持向后兼容
        if self.skip_refinement:
            self.enable_primary_scoring = True  # skip_refinement 时强制仅对 primary 评分
        
        if not self.enable_pq_metric:
            self.logger.info("PQ metric evaluation is DISABLED (enable_pq_metric=False)")
        if not self.enable_sc_metric:
            self.logger.info("SC metric evaluation is DISABLED (enable_sc_metric=False)")
        if self.enable_instruction_following_metric:
            self.logger.info("Instruction Following metric evaluation is ENABLED")
        if self.enable_primary_scoring:
            self.logger.info("Primary image scoring is ENABLED (enable_primary_scoring=True)")
            self.logger.info("  Will evaluate both primary and refined edited images")
        else:
            self.logger.info("Primary image scoring is DISABLED (enable_primary_scoring=False)")
            self.logger.info("  Will only evaluate refined edited images")
        
        # 迭代refinement配置（控制是否启用迭代refinement）
        self.enable_iterative_refinement = config["evaluation"].get("enable_iterative_refinement", False)  # 默认关闭以保持向后兼容
        
        if self.enable_iterative_refinement:
            self.logger.info("Iterative refinement is ENABLED (enable_iterative_refinement=True)")
            self.logger.info("  Multiple re-edit instructions will be processed iteratively")
        else:
            self.logger.info("Iterative refinement is DISABLED (enable_iterative_refinement=False)")
            self.logger.info("  Using single refinement with concatenated re-edit instructions")
        
        # Primary images 目录配置（用于从已有结果加载 primary edited images）
        self.primary_images_dir = config["evaluation"].get("primary_images_dir", None)
        # Primary image 文件名后缀配置（仅在启用 primary_images_dir 时有效）
        self.primary_image_suffix = config["evaluation"].get("primary_image_suffix", "_primary.png")
        # 图片保存/加载时的 category 目录映射（用于兼容不同的目录命名）
        # 当 primary_images_dir 使用中文子目录(物理/环境等)时，可配置 category_to_dir 将英文 category 映射到目录名:
        #   {"physical": "物理", "environmental": "环境", "cultural": "社会", "causal": "因果", "referential": "指代"}
        # 不配置时直接使用 category 作为目录名（英文 subset 时即 physical/environmental 等）
        self.category_to_dir = config["evaluation"].get("category_to_dir", None)
        if self.primary_images_dir:
            self.logger.info(f"Primary images directory: {self.primary_images_dir}")
            self.logger.info(f"Primary image suffix: {self.primary_image_suffix}")
            self.logger.info("Will skip Stage 1 (Primary Editing) and load images from directory")
        
        # 使用原图作为 primary image 的配置（trick：直接使用 original_image 作为 primary_edited_image）
        self.use_original_as_primary = config["evaluation"].get("use_original_as_primary", False)  # 默认关闭
        if self.use_original_as_primary:
            self.logger.info("⚠️  use_original_as_primary is ENABLED: Will use original images directly as primary images")
            self.logger.info("  This means refinement will start from original images instead of primary edited images")
        
        # 跳过 Stage4 (Comparative Scoring) 的配置
        self.skip_stage4 = config["evaluation"].get("skip_stage4", False)  # 默认不跳过
        if self.skip_stage4:
            self.logger.info("⚠️  skip_stage4 is ENABLED: Will skip Stage 4 (Comparative Scoring)")
            self.logger.info("  Pipeline will only perform MLLM analysis and refinement, without scoring")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        if self.enable_disk_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # 创建日志文件路径（实时保存）
        # 如果配置了 use_original_as_primary、primary_images_dir 或 skip_refinement，在日志文件名中添加标识以区分
        if self.skip_refinement:
            log_filename = f"pipeline_{self.primary_model_type}_skip_refinement_{self.start_timestamp}.log"
        elif self.use_original_as_primary:
            log_filename = f"pipeline_{self.primary_model_type}_use_original_as_primary_{self.start_timestamp}.log"
        elif self.primary_images_dir:
            primary_dir_name = os.path.basename(os.path.normpath(self.primary_images_dir))
            log_filename = f"pipeline_{self.primary_model_type}_refined_only_{primary_dir_name}_{self.start_timestamp}.log"
        else:
            log_filename = f"pipeline_{self.primary_model_type}_{self.start_timestamp}.log"
        self.log_file_path = os.path.join(self.output_dir, log_filename)
        
        # 重新配置logger，添加文件输出（实时写入）
        self.logger = setup_logger(
            "iterative_pipeline", 
            config.get("log_level", "INFO"),
            log_file=self.log_file_path,
            console_output=True
        )
        
        # 评分健康度追踪（用于记录GPU失败信息）
        self.scoring_health = {
            'primary_failures': {},      # Primary评分失败信息（仅在enable_primary_scoring时使用）
            'refined_failures': {},       # Refined评分失败信息
            'total_primary_samples': 0,   # 总primary样本数
            'total_primary_failures': 0,  # 总primary失败数
            'total_refined_samples': 0,   # 总refined样本数
            'total_refined_failures': 0   # 总refined失败数
        }
        
        self.logger.info("Iterative Refinement Pipeline initialized successfully (models will be loaded on demand)")
        self.logger.info(f"Log file: {self.log_file_path}")
    
    def _ensure_model_loaded(self, model_name: str):
        """
        确保指定模型已加载，并自动卸载之前的模型
        
        Args:
            model_name: 模型名称 ("primary_diffusion", "refinement_diffusion", "mllm", "reward_model")
        """
        # 如果当前已经是要加载的模型，直接返回
        if self._current_model_name == model_name and self._current_loaded_model is not None:
            self.logger.info(f"[Model Manager] {model_name} already loaded, skipping")
            return
        
        # 先卸载当前模型
        if self._current_loaded_model is not None:
            self.logger.info(f"[Model Manager] Unloading {self._current_model_name}...")
            if hasattr(self._current_loaded_model, 'unload_from_gpu'):
                self._current_loaded_model.unload_from_gpu()
            # 完全释放实例（可选，取决于是否需要重用）
            # del self._current_loaded_model
            # self._current_loaded_model = None
            self._current_loaded_model = None
            self._current_model_name = None
        
        # 加载新模型
        self.logger.info(f"[Model Manager] Loading {model_name}...")
        
        if model_name == "primary_diffusion":
            if self.primary_diffusion is None:
                self.primary_diffusion = self._load_primary_diffusion()
            else:
                if hasattr(self.primary_diffusion, 'load_to_gpu'):
                    self.primary_diffusion.load_to_gpu()
            self._current_loaded_model = self.primary_diffusion
        
        elif model_name == "refinement_diffusion":
            if self.refinement_diffusion is None:
                self.refinement_diffusion = self._load_refinement_diffusion()
            else:
                if hasattr(self.refinement_diffusion, 'load_to_gpu'):
                    self.refinement_diffusion.load_to_gpu()
            self._current_loaded_model = self.refinement_diffusion
        
        elif model_name == "mllm":
            if self.mllm is None:
                self.mllm = self._load_mllm()
            else:
                if hasattr(self.mllm, 'load_to_gpu'):
                    self.mllm.load_to_gpu()
            self._current_loaded_model = self.mllm
        
        elif model_name == "reward_model":
            if self.reward_model is None:
                self.reward_model = self._load_reward_model()
            else:
                if hasattr(self.reward_model, 'load_to_gpu'):
                    self.reward_model.load_to_gpu()
            self._current_loaded_model = self.reward_model
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        self._current_model_name = model_name
        self.logger.info(f"[Model Manager] {model_name} loaded and ready")
    
    def _unload_current_model(self):
        """卸载当前加载的模型"""
        if self._current_loaded_model is not None:
            self.logger.info(f"[Model Manager] Unloading {self._current_model_name}...")
            if hasattr(self._current_loaded_model, 'unload_from_gpu'):
                self._current_loaded_model.unload_from_gpu()
            self._current_loaded_model = None
            self._current_model_name = None
            self.logger.info(f"[Model Manager] Model unloaded")
    
    def _load_primary_diffusion(self):
        """
        加载待评测Diffusion模型（初次编辑）
        
        Returns:
            Diffusion模型实例
        """
        diffusion_config = self.config["diffusion_model"]["primary"]
        model_type = diffusion_config["type"]
        model_params = diffusion_config.get("params", {})
        
        self.logger.info(f"Loading primary diffusion model: {model_type}")
        
        # 动态导入
        if model_type == "multi_gpu_qwen_edit":
            from .models.diffusion.implementations.qwen_image_edit import QwenImageEditModel
            return QwenImageEditModel(model_params)
        elif model_type == "qwen_image_edit_2511":
            from .models.diffusion.implementations.qwen_image_edit_2511 import QwenImageEdit2511Model
            return QwenImageEdit2511Model(model_params)
        elif model_type == "flux_kontext":
            from .models.diffusion.implementations.flux_kontext import FluxKontextModel
            return FluxKontextModel(model_params)
        elif model_type == "step1x_edit_v1p2_preview":
            from .models.diffusion.implementations.step1x_edit_v1p2_preview import Step1XEditModel
            return Step1XEditModel(model_params)
        elif model_type == "step1x_edit_v1p1":
            from .models.diffusion.implementations.step1x_edit_v1p1 import Step1XEditPreviewModel
            return Step1XEditPreviewModel(model_params)
        elif model_type == "dreamomni2":
            from .models.diffusion.implementations.dreamomni2 import DreamOmni2Model
            return DreamOmni2Model(model_params)
        elif model_type == "omnigen2":
            from .models.diffusion.implementations.omnigen2 import OmniGen2Model
            return OmniGen2Model(model_params)
        elif model_type == "hidream_e1":
            from .models.diffusion.implementations.hidream_e1 import HiDreamE1Model
            return HiDreamE1Model(model_params)
        elif model_type == "ovis_u1":
            from .models.diffusion.implementations.ovis_u1 import OvisU1Model
            return OvisU1Model(model_params)
        elif model_type == "janus":
            from .models.diffusion.implementations.janus import JanusModel
            return JanusModel(model_params)
        elif model_type == "flux2_dev":
            from .models.diffusion.implementations.flux2_dev import Flux2DevModel
            return Flux2DevModel(model_params)
        else:
            raise ValueError(f"Unknown primary diffusion model type: {model_type}")
    
    def _load_refinement_diffusion(self):
        """
        加载二次编辑Diffusion模型（固定模组）
        
        这个模型与MLLM构成固定的优化模组，用于基于MLLM生成的Re-edit指令
        对初次编辑结果进行二次优化。通常使用标准的Qwen-Image-Edit模型。
        
        Returns:
            Diffusion模型实例
        """
        refinement_config = self.config["diffusion_model"].get("refinement")
        
        if not refinement_config:
            self.logger.warning("No refinement diffusion model configured, using primary model as fallback")
            return None
        
        model_type = refinement_config["type"]
        model_params = refinement_config.get("params", {})
        
        self.logger.info(f"Loading refinement diffusion model: {model_type}")
        self.logger.info(f"  Model: {model_params.get('model_name', 'N/A')}")
        self.logger.info(f"  GPUs: {model_params.get('device_ids', 'auto')}")
        
        # 动态导入
        if model_type == "multi_gpu_qwen_edit":
            from .models.diffusion.implementations.qwen_image_edit import QwenImageEditModel
            return QwenImageEditModel(model_params)
        elif model_type == "qwen_image_edit_2511":
            from .models.diffusion.implementations.qwen_image_edit_2511 import QwenImageEdit2511Model
            return QwenImageEdit2511Model(model_params)
        elif model_type == "flux_kontext":
            from .models.diffusion.implementations.flux_kontext import FluxKontextModel
            return FluxKontextModel(model_params)
        elif model_type == "step1x_edit_v1p2_preview":
            from .models.diffusion.implementations.step1x_edit_v1p2_preview import Step1XEditModel
            return Step1XEditModel(model_params)
        elif model_type == "step1x_edit_v1p1":
            from .models.diffusion.implementations.step1x_edit_v1p1 import Step1XEditPreviewModel
            return Step1XEditPreviewModel(model_params)
        elif model_type == "dreamomni2":
            from .models.diffusion.implementations.dreamomni2 import DreamOmni2Model
            return DreamOmni2Model(model_params)
        elif model_type == "omnigen2":
            from .models.diffusion.implementations.omnigen2 import OmniGen2Model
            return OmniGen2Model(model_params)
        elif model_type == "hidream_e1":
            from .models.diffusion.implementations.hidream_e1 import HiDreamE1Model
            return HiDreamE1Model(model_params)
        elif model_type == "ovis_u1":
            from .models.diffusion.implementations.ovis_u1 import OvisU1Model
            return OvisU1Model(model_params)
        elif model_type == "janus":
            from .models.diffusion.implementations.janus import JanusModel
            return JanusModel(model_params)
        elif model_type == "flux2_dev":
            from .models.diffusion.implementations.flux2_dev import Flux2DevModel
            return Flux2DevModel(model_params)
        else:
            raise ValueError(f"Unknown refinement diffusion model type: {model_type}")
    
    def _load_mllm(self):
        """
        加载MLLM模型
        
        Returns:
            MLLM模型实例
        """
        mllm_config = self.config["mllm"]
        model_type = mllm_config["type"]
        model_params = mllm_config.get("params", {})
        
        self.logger.info(f"Loading MLLM model: {model_type}")
        
        # 动态导入
        if model_type == "qwen25_vl":
            from .models.mllm.implementations.qwen25_vl_mllm import Qwen25VLMllm
            return Qwen25VLMllm(model_params)
        else:
            raise ValueError(f"Unknown MLLM type: {model_type}")
    
    def _load_reward_model(self):
        """
        加载Reward模型
        
        Returns:
            Reward模型实例
        """
        reward_config = self.config["reward_model"]
        model_type = reward_config["type"]
        model_params = reward_config.get("params", {})
        
        self.logger.info(f"Loading reward model: {model_type}")
        
        # 传递日志文件路径给reward model（用于子进程输出）
        if hasattr(self, 'log_file_path'):
            model_params = model_params.copy()  # 避免修改原始配置
            model_params['log_file_path'] = self.log_file_path
        
        # 动态导入
        if model_type == "qwen3_vl_multi_gpu_subprocess":
            from .models.reward.implementations.qwen3_vl_multi_gpu_subprocess import Qwen3VLMultiGPUSubprocessRewardModel
            return Qwen3VLMultiGPUSubprocessRewardModel(model_params)
        elif model_type == "qwen3_vl_vllm_subprocess":
            from .models.reward.implementations.qwen3_vl_vllm_subprocess import Qwen3VLvLLMSubprocessRewardModel
            return Qwen3VLvLLMSubprocessRewardModel(model_params)
        elif model_type == "qwen3_vl_subprocess":
            from .models.reward.implementations.qwen3_vl_subprocess import Qwen3VLSubprocessRewardModel
            return Qwen3VLSubprocessRewardModel(model_params)
        elif model_type == "qwen3_vl":
            from .models.reward.implementations.qwen3_vl_reward import Qwen3VLRewardModel
            return Qwen3VLRewardModel(model_params)
        else:
            raise ValueError(f"Unknown reward model type: {model_type}")
    
    def _load_benchmark_data(self):
        """
        加载benchmark数据
        
        与standard pipeline保持一致的数据加载逻辑
        
        Returns:
            BenchmarkData对象
        """
        # 从配置中读取数据路径和类别
        data_config = self.config.get("data", {})
        data_path = data_config.get("path")
        categories = data_config.get("categories", [])
        
        if not data_path:
            raise ValueError("data.path not specified in config")
        
        if not categories:
            raise ValueError("data.categories not specified in config")
        
        self.logger.info(f"Loading benchmark data from: {data_path}")
        self.logger.info(f"Categories: {categories}")
        
        # 调用data_loader.load()
        benchmark_data = self.data_loader.load(
            data_path=data_path,
            categories=categories,
            decode_images=False  # 延迟解码以节省内存
        )
        
        self.logger.info(f"Loaded {benchmark_data.total_pairs} data pairs across {len(categories)} categories")
        
        return benchmark_data
    
    def run(self) -> Dict[str, Any]:
        """
        运行完整的迭代优化评测
        
        Returns:
            评测报告字典（与standard pipeline兼容）
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Iterative Refinement Benchmark")
        self.logger.info("=" * 80)
        
        # 加载数据
        benchmark_data = self._load_benchmark_data()
        
        # 转换为IterativeDataPair
        iterative_benchmark = self._convert_to_iterative_data(benchmark_data)
        
        # 按类别处理
        for category_name in iterative_benchmark.category_names:
            category_data = iterative_benchmark.get_category(category_name)
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Processing category: {category_name}")
            self.logger.info(f"{'='*80}")
            
            self._process_category_iterative(category_name, category_data)
        
        # 生成报告
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Generating comparison report...")
        self.logger.info("=" * 80)
        report = self._generate_report(iterative_benchmark)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✓ Iterative Refinement Benchmark completed successfully!")
        self.logger.info("=" * 80)
        
        return report
    
    def _convert_to_iterative_data(self, benchmark_data) -> IterativeBenchmarkData:
        """
        将标准BenchmarkData转换为IterativeBenchmarkData
        
        Args:
            benchmark_data: 标准BenchmarkData
        
        Returns:
            IterativeBenchmarkData
        """
        iterative_categories = {}
        
        for category_name, category_data in benchmark_data.categories.items():
            # 转换数据对
            iterative_pairs = []
            for pair in category_data.data_pairs:
                iterative_pair = IterativeDataPair(
                    pair_id=pair.pair_id,
                    category=pair.category,
                    original_image_b64=pair.original_image_b64,
                    edit_instruction=pair.edit_instruction,
                    original_description=pair.original_description,
                    original_image=pair.original_image,
                    rationale=pair.rationale,  # 修复：传递rationale字段
                    metadata=pair.metadata
                )
                iterative_pairs.append(iterative_pair)
            
            # 创建类别数据
            iterative_categories[category_name] = IterativeCategoryData(
                category_name=category_name,
                data_pairs=iterative_pairs
            )
        
        return IterativeBenchmarkData(
            categories=iterative_categories,
            total_pairs=benchmark_data.total_pairs,
            category_names=benchmark_data.category_names,
            metadata=benchmark_data.metadata
        )
    
    def _process_category_iterative(self, category: str, category_data: IterativeCategoryData):
        """
        迭代优化处理单个类别
        
        Args:
            category: 类别名称
            category_data: 类别数据
        """
        # 计算总阶段数（动态调整）
        # skip_refinement: 2或3阶段（Primary/加载, Scoring, Statistics）
        # 如果跳过 Stage1：基础3个阶段（MLLM, Refinement, Statistics），如果执行 Stage4 则+1
        # 如果执行 Stage1：基础5个阶段（Primary, MLLM, Refinement, Scoring, Statistics），如果跳过 Stage4 则-1
        if self.skip_refinement:
            total_stages = 2 if self.skip_stage4 else 3  # Primary/加载, (Scoring), Statistics
        elif self.use_original_as_primary or self.primary_images_dir:
            total_stages = 3 if self.skip_stage4 else 4
        else:
            total_stages = 4 if self.skip_stage4 else 5
        
        # ===== skip_refinement 模式：Primary -> Scoring -> Statistics =====
        if self.skip_refinement:
            if self.use_original_as_primary:
                self.logger.info(f"\n[STAGE 1/{total_stages}] Using Original Images as Primary Images ({category})")
                self._use_original_as_primary(category, category_data)
            elif self.primary_images_dir:
                self.logger.info(f"\n[STAGE 1/{total_stages}] Loading Primary Images from Directory ({category})")
                self._load_primary_images_from_dir(category, category_data)
            else:
                self.logger.info(f"\n[STAGE 1/{total_stages}] Primary Editing ({category})")
                self._stage1_primary_editing(category, category_data)
            
            if not self.skip_stage4:
                self.logger.info(f"\n[STAGE 2/{total_stages}] Primary-Only Scoring ({category})")
                self._stage4_comparative_scoring(category, category_data)
                self.logger.info(f"\n[STAGE 3/{total_stages}] Statistics ({category})")
                self._stage5_statistics(category, category_data)
            else:
                self.logger.info(f"\n[SKIPPED] Stage 2 (Scoring) and Stage 3 (Statistics) are skipped")
            return
        
        # ===== 阶段1: 加载已有 Primary Images =====
        if self.use_original_as_primary:
            # Trick: 直接使用原图作为 primary image
            self.logger.info(f"\n[STAGE 1/{total_stages}] Using Original Images as Primary Images ({category})")
            self._use_original_as_primary(category, category_data)
        elif self.primary_images_dir:
            self.logger.info(f"\n[STAGE 1/{total_stages}] Loading Primary Images from Directory ({category})")
            self._load_primary_images_from_dir(category, category_data)
        else:
            # 如果没有配置 primary_images_dir，仍然执行 stage1（向后兼容）
            self.logger.info(f"\n[STAGE 1/{total_stages}] Primary Editing ({category})")
            self._stage1_primary_editing(category, category_data)
            # 如果执行了 stage1，继续后续阶段
            self.logger.info(f"\n[STAGE 2/{total_stages}] MLLM Analysis ({category})")
            self._stage2_mllm_analysis(category, category_data)
            self.logger.info(f"\n[STAGE 3/{total_stages}] Refinement Editing ({category})")
            self._stage3_refinement_editing(category, category_data)
            
            # 根据 skip_stage4 决定是否执行 Stage4
            if not self.skip_stage4:
                self.logger.info(f"\n[STAGE 4/{total_stages}] Comparative Scoring ({category})")
                self._stage4_comparative_scoring(category, category_data)
                # 如果执行了 Stage4，执行 Stage5 统计
                self.logger.info(f"\n[STAGE 5/{total_stages}] Statistics ({category})")
                self._stage5_statistics(category, category_data)
            else:
                # 如果跳过 Stage4，跳过 Stage5 统计（或只做基础统计）
                self.logger.info(f"\n[SKIPPED] Stage 4 (Comparative Scoring) and Stage 5 (Statistics) are skipped")
            
            return
        
        # ===== 阶段2: MLLM分析 (MLLM Analysis) =====
        self.logger.info(f"\n[STAGE 2/{total_stages}] MLLM Analysis ({category})")
        self._stage2_mllm_analysis(category, category_data)
        
        # ===== 阶段3: 二次编辑 (Refinement Editing) =====
        self.logger.info(f"\n[STAGE 3/{total_stages}] Refinement Editing ({category})")
        self._stage3_refinement_editing(category, category_data)
        
        # ===== 阶段4: 评分 (Scoring) =====
        if not self.skip_stage4:
            self.logger.info(f"\n[STAGE 4/{total_stages}] Scoring ({category})")
            self._stage4_comparative_scoring(category, category_data)
            
            # ===== 统计记录 (Statistics) =====
            self.logger.info(f"\n[STATISTICS] Statistics ({category})")
            self._stage5_statistics(category, category_data)
        else:
            # 如果跳过 Stage4，跳过 Stage5 统计
            self.logger.info(f"\n[SKIPPED] Stage 4 (Comparative Scoring) and Stage 5 (Statistics) are skipped")
    
    def _stage1_primary_editing(self, category: str, category_data: IterativeCategoryData):
        """
        阶段1: 初次编辑
        
        使用待评测Diffusion模型进行初次编辑
        """
        # 加载Primary Diffusion模型
        self._ensure_model_loaded("primary_diffusion")
        
        # 准备数据
        original_images = []
        edit_instructions = []
        
        for pair in category_data.data_pairs:
            if pair.original_image is None:
                pair.original_image = decode_base64_image(pair.original_image_b64)
            original_images.append(pair.original_image)
            edit_instructions.append(pair.edit_instruction)
        
        # 批量编辑
        self.logger.info(f"Editing {len(original_images)} images with primary model...")
        
        edited_images = []
        if hasattr(self.primary_diffusion, 'batch_edit'):
            edited_images = self.primary_diffusion.batch_edit(
                images=original_images,
                instructions=edit_instructions
            )
        else:
            # Fallback to sequential
            self.logger.warning("Primary model does not support batch_edit, using sequential processing")
            for i, (img, instr) in enumerate(zip(original_images, edit_instructions)):
                edited = self.primary_diffusion.edit_image(img, instr)
                edited_images.append(edited)
        
        # 保存结果
        for pair, edited_image in zip(category_data.data_pairs, edited_images):
            pair.primary_edited_image = edited_image
            
            # 保存图像（可选）
            if self.save_images:
                self._save_image(edited_image, category, pair.pair_id, "primary")
        
        self.logger.info(f"Primary editing completed for {len(edited_images)} images")
        
        # 卸载模型
        self._unload_current_model()
    
    def _stage2_mllm_analysis(self, category: str, category_data: IterativeCategoryData):
        """
        阶段2: MLLM分析
        
        使用MLLM分析初次编辑结果，生成改进指令
        """
        # 加载MLLM模型
        self._ensure_model_loaded("mllm")
        
        # 准备数据
        # 确保 primary_edited_image 和 original_image 都已加载
        edited_images = []
        original_images = []
        original_descriptions = []
        edit_instructions = []
        
        for pair in category_data.data_pairs:
            # 检查 primary_edited_image
            if pair.primary_edited_image is None:
                raise RuntimeError(
                    f"primary_edited_image is None for pair {pair.pair_id} in category {category}. "
                    f"Cannot proceed with MLLM analysis."
                )
            
            # 确保 original_image 已解码（如果使用 primary_images_dir，可能还未解码）
            if pair.original_image is None:
                if pair.original_image_b64:
                    pair.original_image = decode_base64_image(pair.original_image_b64)
                else:
                    raise RuntimeError(
                        f"original_image and original_image_b64 are both None for pair {pair.pair_id} "
                        f"in category {category}. Cannot proceed with MLLM analysis."
                    )
            
            edited_images.append(pair.primary_edited_image)
            original_images.append(pair.original_image)
            original_descriptions.append(pair.original_description if pair.original_description else "")
            edit_instructions.append(pair.edit_instruction if pair.edit_instruction else "")
        
        categories = [category] * len(category_data.data_pairs)
        
        # 定义实时输出回调函数
        def on_batch_complete(batch_idx, batch_results, batch_indices):
            """在每个batch完成后立即输出结果"""
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"MLLM Batch {batch_idx + 1} Results:")
            self.logger.info(f"{'='*60}")
            
            for idx, output in zip(batch_indices, batch_results):
                pair = category_data.data_pairs[idx]
                
                # 立即保存结果到数据对象
                pair.cot_reasoning = output.get("cot", "")
                pair.re_edit_instruction = output.get("re_edit_instruction", "")  # 拼接后的字符串（向后兼容）
                pair.re_edit_instructions = output.get("re_edit_instructions", [])  # 独立的指令列表（用于迭代refinement）
                pair.mllm_raw_output = output.get("raw_output", "")
                
                # 实时输出每个样本的MLLM分析结果
                self.logger.info(f"\n[Sample {pair.pair_id}]")
                self.logger.info(f"  Original Instruction: {pair.edit_instruction[:80]}...")
                if pair.cot_reasoning:
                    self.logger.info(f"  CoT Reasoning: {pair.cot_reasoning[:150]}...")
                else:
                    self.logger.info(f"  CoT Reasoning: (empty)")
                self.logger.info(f"  Re-edit Instruction: {pair.re_edit_instruction[:100]}...")
        
        # 批量分析（带实时输出）
        self.logger.info(f"Analyzing {len(edited_images)} images with MLLM...")
        
        mllm_outputs = self.mllm.batch_analyze(
            edited_images=edited_images,
            original_images=original_images,
            original_descriptions=original_descriptions,
            edit_instructions=edit_instructions,
            categories=categories,
            on_batch_complete=on_batch_complete  # 传入回调函数
        )
        
        # 确保所有结果都已保存（防止回调失败的情况）
        for pair, output in zip(category_data.data_pairs, mllm_outputs):
            if not hasattr(pair, 'cot_reasoning') or pair.cot_reasoning is None:
                pair.cot_reasoning = output.get("cot", "")
            if not hasattr(pair, 're_edit_instruction') or pair.re_edit_instruction is None:
                pair.re_edit_instruction = output.get("re_edit_instruction", "")  # 拼接后的字符串（向后兼容）
            if not hasattr(pair, 're_edit_instructions') or pair.re_edit_instructions is None:
                pair.re_edit_instructions = output.get("re_edit_instructions", [])  # 独立的指令列表（用于迭代refinement）
            if not hasattr(pair, 'mllm_raw_output') or pair.mllm_raw_output is None:
                pair.mllm_raw_output = output.get("raw_output", "")
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"MLLM analysis completed for {len(mllm_outputs)} images")
        self.logger.info(f"{'='*60}")
        
        # 卸载模型
        self._unload_current_model()
    
    def _stage3_refinement_editing(self, category: str, category_data: IterativeCategoryData):
        """
        阶段3: 二次编辑
        
        使用Re-edit指令进行二次编辑
        支持迭代refinement：如果启用且MLLM输出多个指令，则进行多次迭代refinement
        """
        # ===== 数据完整性验证 =====
        # 检查Stage 1的输出是否完整
        for pair in category_data.data_pairs:
            if pair.primary_edited_image is None:
                raise RuntimeError(
                    f"Pipeline implementation error: primary_edited_image is None for pair {pair.pair_id} "
                    f"in category {category}. This indicates Stage 1 (Primary Editing) failed to produce output."
                )
            if not pair.re_edit_instruction or len(pair.re_edit_instruction.strip()) == 0:
                raise RuntimeError(
                    f"Pipeline implementation error: re_edit_instruction is empty for pair {pair.pair_id} "
                    f"in category {category}. This indicates Stage 2 (MLLM Analysis) failed to produce valid output."
                )
        
        self.logger.info(f"Data integrity check passed: all {len(category_data.data_pairs)} pairs have valid primary_edited_image and re_edit_instruction")
        
        # 加载Refinement Diffusion模型（如果配置了独立模型）或重用Primary模型
        if self.refinement_diffusion is not None or self.config["diffusion_model"].get("refinement"):
            self._ensure_model_loaded("refinement_diffusion")
            refinement_model = self.refinement_diffusion
        else:
            # 如果没有独立的refinement模型，重用primary模型
            self._ensure_model_loaded("primary_diffusion")
            refinement_model = self.primary_diffusion
            self.logger.info("Using primary_diffusion for refinement (no separate refinement model configured)")
        
        # ===== 判断是否使用迭代refinement =====
        use_iterative = self.enable_iterative_refinement
        
        if use_iterative:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"[Iterative Refinement Mode] ENABLED")
            self.logger.info(f"{'='*80}")
        else:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"[Single Refinement Mode] Using concatenated re-edit instructions")
            self.logger.info(f"{'='*80}")
        
        # ===== 迭代refinement逻辑 =====
        if use_iterative:
            # 迭代refinement模式：多GPU并行处理，每个GPU内部串行迭代
            self.logger.info(f"Processing {len(category_data.data_pairs)} pairs with iterative refinement (multi-GPU parallel)...")
            print(f"🔄 Processing {len(category_data.data_pairs)} pairs with iterative refinement (multi-GPU parallel)...")
            
            # 统计每个pair的迭代次数并打印
            total_iterations = 0
            for pair in category_data.data_pairs:
                if pair.re_edit_instructions:
                    iterations = len(pair.re_edit_instructions)
                else:
                    iterations = 1 if pair.re_edit_instruction else 0
                total_iterations += iterations
                pair_info = f"  📊 Pair {pair.pair_id}: {iterations} iteration(s)"
                self.logger.info(pair_info)
                print(pair_info)
            
            summary_info = f"  📈 Total: {len(category_data.data_pairs)} pairs, {total_iterations} total iterations"
            self.logger.info(summary_info)
            print(summary_info)
            
            # 获取GPU设备信息
            device_ids, num_gpus = self._get_gpu_info(refinement_model)
            gpu_info = f"Using {num_gpus} GPUs: {device_ids}"
            self.logger.info(gpu_info)
            print(f"🖥️  {gpu_info}")
            
            # 任务分配：将pairs随机分配到不同GPU
            gpu_tasks = self._assign_pairs_to_gpus(category_data.data_pairs, device_ids, num_gpus)
            
            # 并行执行：每个GPU处理分配给它的pairs
            refined_images = self._parallel_iterative_refinement(
                refinement_model, gpu_tasks, device_ids, category, category_data
            )
            
            # 保存图像（可选）
            if self.save_images:
                for pair, refined_image in zip(category_data.data_pairs, refined_images):
                    self._save_image(refined_image, category, pair.pair_id, "refined")
            
            # 统计信息
            total_iterations = sum(pair.refinement_iterations or 0 for pair in category_data.data_pairs)
            avg_iterations = total_iterations / len(category_data.data_pairs) if category_data.data_pairs else 0
            max_iterations = max((pair.refinement_iterations or 0 for pair in category_data.data_pairs), default=0)
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"[Iterative Refinement Summary]")
            self.logger.info(f"  Total pairs: {len(category_data.data_pairs)}")
            self.logger.info(f"  Total iterations: {total_iterations}")
            self.logger.info(f"  Average iterations per pair: {avg_iterations:.2f}")
            self.logger.info(f"  Max iterations: {max_iterations}")
            self.logger.info(f"{'='*80}")
        
        else:
            # ===== 原有的一次refinement逻辑（向后兼容）=====
            # 准备数据（二次编辑以初次编辑结果为基础）
            base_images = [pair.primary_edited_image for pair in category_data.data_pairs]
            
            # 仅使用 re_edit_instruction（不拼接CoT）
            re_edit_instructions = [pair.re_edit_instruction for pair in category_data.data_pairs]
            
            # 批量编辑（使用与初次编辑相同的多GPU并行机制）
            self.logger.info(f"Re-editing {len(base_images)} images with refinement model...")
            self.logger.info(f"  Using re-edit instructions only (CoT not included)")
            
            refined_images = []
            if hasattr(refinement_model, 'batch_edit'):
                # 优先使用batch_edit进行多GPU并行编辑
                refined_images = refinement_model.batch_edit(
                    images=base_images,
                    instructions=re_edit_instructions
                )
            else:
                # Fallback: 串行编辑
                self.logger.warning("Refinement model does not support batch_edit, falling back to sequential processing")
                for i, (img, instr) in enumerate(zip(base_images, re_edit_instructions)):
                    self.logger.info(f"  Re-editing image {i+1}/{len(base_images)}")
                    refined = refinement_model.edit_image(img, instr)
                    refined_images.append(refined)
            
            # 保存结果
            for pair, refined_image in zip(category_data.data_pairs, refined_images):
                pair.refined_edited_image = refined_image
                pair.refinement_iterations = 1  # 单次refinement记录为1次迭代
            
            # 保存图像（可选）
            if self.save_images:
                for pair, refined_image in zip(category_data.data_pairs, refined_images):
                    self._save_image(refined_image, category, pair.pair_id, "refined")
        
        self.logger.info(f"Refinement editing completed for {len(refined_images)} images")
        
        # 卸载模型
        self._unload_current_model()
    
    def _stage4_comparative_scoring(self, category: str, category_data: IterativeCategoryData):
        """
        阶段4: 评分
        
        skip_refinement 时：仅对 primary 评分，refined_* 保持 None
        否则：根据 enable_primary_scoring 决定是否评分 primary，总是评分 refined
        """
        # ===== 数据完整性验证 =====
        if self.skip_refinement:
            # skip_refinement 时只检查 primary_edited_image
            for pair in category_data.data_pairs:
                if pair.primary_edited_image is None:
                    raise RuntimeError(
                        f"Pipeline implementation error: primary_edited_image is None for pair {pair.pair_id} "
                        f"in category {category}. skip_refinement mode requires valid primary images."
                    )
            self.logger.info(f"Data integrity check passed (skip_refinement): all {len(category_data.data_pairs)} pairs have valid primary images")
        else:
            # 正常模式：检查 refined，可选检查 primary
            for pair in category_data.data_pairs:
                if pair.refined_edited_image is None:
                    raise RuntimeError(
                        f"Pipeline implementation error: refined_edited_image is None for pair {pair.pair_id} "
                        f"in category {category}. This indicates Stage 3 (Refinement Editing) failed to produce output."
                    )
                if self.enable_primary_scoring and pair.primary_edited_image is None:
                    raise RuntimeError(
                        f"Pipeline implementation error: primary_edited_image is None for pair {pair.pair_id} "
                        f"in category {category}. This indicates Stage 1 (Primary Editing) failed to produce output."
                    )
            if self.enable_primary_scoring:
                self.logger.info(f"Data integrity check passed: all {len(category_data.data_pairs)} pairs have valid primary and refined images")
            else:
                self.logger.info(f"Data integrity check passed: all {len(category_data.data_pairs)} pairs have valid refined images")
        
        # 加载Reward模型
        self._ensure_model_loaded("reward_model")
        
        # 准备通用数据
        original_images = [pair.original_image for pair in category_data.data_pairs]
        original_descriptions = [pair.original_description for pair in category_data.data_pairs]
        edit_instructions = [pair.edit_instruction for pair in category_data.data_pairs]
        rationales = [pair.rationale for pair in category_data.data_pairs]  # 新增：提取rationale
        
        # 打印rationales传递情况（用于调试和记录）
        rationales_with_value = sum(1 for r in rationales if r and len(str(r).strip()) > 0)
        rationales_none_or_empty = len(rationales) - rationales_with_value
        self.logger.info(f"[Rationale Check] Total pairs: {len(rationales)}, "
                        f"With rationale: {rationales_with_value}, "
                        f"None/Empty: {rationales_none_or_empty}")
        if rationales_with_value > 0:
            # 显示前3个有值的rationale示例
            sample_rationales = [r for r in rationales if r and len(str(r).strip()) > 0][:3]
            for i, r in enumerate(sample_rationales, 1):
                self.logger.info(f"  Sample rationale {i}: {str(r)[:100]}...")
        
        # 获取prompts
        system_prompts = [self.prompt_manager.get_system_prompt(category)] * len(category_data.data_pairs)
        user_prompts = [
            self.prompt_manager.get_user_prompt(category, pair.original_description, pair.edit_instruction)
            for pair in category_data.data_pairs
        ]
        
        # ===== 类别特定评价：Primary编辑结果 =====
        # skip_refinement 时 enable_primary_scoring 恒为 True，只评 primary
        primary_images = [pair.primary_edited_image for pair in category_data.data_pairs]
        if self.enable_primary_scoring:
            self.logger.info(f"Scoring primary edited images...")
            
            primary_scores, primary_reasonings, primary_failures = self.reward_model.batch_score(
                edited_images=primary_images,
                original_descriptions=original_descriptions,
                edit_instructions=edit_instructions,
                system_prompts=system_prompts,
                user_prompts=user_prompts,
                original_images=original_images,
                rationales=rationales,
                return_failures=True
            )
            
            # 记录primary评分失败信息
            self.scoring_health['primary_failures'][category] = primary_failures
            self.scoring_health['total_primary_samples'] += primary_failures['total_samples']
            self.scoring_health['total_primary_failures'] += primary_failures['failed_sample_count']
            
            # 保存primary分数和reasoning
            for pair, primary_score, primary_reasoning in zip(
                category_data.data_pairs, primary_scores, primary_reasonings
            ):
                pair.primary_score = primary_score  # "yes" or "no"
                pair.primary_score_reasoning = primary_reasoning  # reasoning文本
            
            # 统计primary yes/no结果
            primary_yes_count = sum(1 for s in primary_scores if s and isinstance(s, str) and s.lower() == "yes")
            primary_no_count = len(primary_scores) - primary_yes_count
            
            self.logger.info(f"Primary scoring completed for {len(primary_scores)} images")
            self.logger.info(f"  Primary: yes={primary_yes_count}, no={primary_no_count}")
        
        # ===== 类别特定评价：Refined编辑结果（skip_refinement 时跳过）=====
        if not self.skip_refinement:
            self.logger.info(f"Scoring refined edited images...")
            refined_images = [pair.refined_edited_image for pair in category_data.data_pairs]
            
            refined_scores, refined_reasonings, refined_failures = self.reward_model.batch_score(
                edited_images=refined_images,
                original_descriptions=original_descriptions,
                edit_instructions=edit_instructions,
                system_prompts=system_prompts,
                user_prompts=user_prompts,
                original_images=original_images,
                rationales=rationales,
                return_failures=True
            )
            
            self.scoring_health['refined_failures'][category] = refined_failures
            self.scoring_health['total_refined_samples'] += refined_failures['total_samples']
            self.scoring_health['total_refined_failures'] += refined_failures['failed_sample_count']
            
            for pair, refined_score, refined_reasoning in zip(
                category_data.data_pairs, refined_scores, refined_reasonings
            ):
                pair.refined_score = refined_score
                pair.refined_score_reasoning = refined_reasoning
            
            refined_yes_count = sum(1 for s in refined_scores if s and isinstance(s, str) and s.lower() == "yes")
            refined_no_count = len(refined_scores) - refined_yes_count
            self.logger.info(f"Refined scoring completed for {len(refined_scores)} images")
            self.logger.info(f"  Refined: yes={refined_yes_count}, no={refined_no_count}")
            
            # 计算improvement rate（仅当有refined时）
            for pair in category_data.data_pairs:
                pair.calculate_improvement_rate()
        
        # ===== PQ指标评价 =====
        if self.enable_pq_metric:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"[PQ Metric Evaluation] Starting PQ metric evaluation for {category}")
            self.logger.info(f"{'='*80}")
            
            # 确保reward model已加载（如果之前卸载了）
            self._ensure_model_loaded("reward_model")
            
            # PQ指标评价 - Primary
            if self.enable_primary_scoring:
                self.logger.info(f"\n[PQ Metric] Scoring primary edited images...")
                primary_pq_scores, primary_pq_reasonings, primary_pq_failures = self._score_pq_metric(
                    edited_images=primary_images,
                    edit_instructions=edit_instructions,
                    original_images=original_images,
                    return_failures=True
                )
                for pair, primary_pq_score, primary_pq_reasoning in zip(
                    category_data.data_pairs, primary_pq_scores, primary_pq_reasonings
                ):
                    pair.primary_pq_score = primary_pq_score
                    pair.primary_pq_reasoning = primary_pq_reasoning
            
            # PQ指标评价 - Refined（skip_refinement 时跳过）
            if not self.skip_refinement:
                self.logger.info(f"\n[PQ Metric] Scoring refined edited images...")
                refined_pq_scores, refined_pq_reasonings, refined_pq_failures = self._score_pq_metric(
                    edited_images=[p.refined_edited_image for p in category_data.data_pairs],
                    edit_instructions=edit_instructions,
                    original_images=original_images,
                    return_failures=True
                )
                for pair, refined_pq_score, refined_pq_reasoning in zip(
                    category_data.data_pairs, refined_pq_scores, refined_pq_reasonings
                ):
                    pair.refined_pq_score = refined_pq_score
                    pair.refined_pq_reasoning = refined_pq_reasoning
            
            # 计算PQ指标统计
            if self.skip_refinement and primary_pq_scores:
                avg_primary_naturalness = sum(s[0] for s in primary_pq_scores) / len(primary_pq_scores)
                avg_primary_artifacts = sum(s[1] for s in primary_pq_scores) / len(primary_pq_scores)
                self.logger.info(f"\n[PQ Metric] Evaluation completed (skip_refinement)! Primary - Naturalness: {avg_primary_naturalness:.2f}, Artifacts: {avg_primary_artifacts:.2f}")
            elif self.enable_primary_scoring and primary_pq_scores and not self.skip_refinement:
                refined_pq_scores = [p.refined_pq_score for p in category_data.data_pairs if p.refined_pq_score is not None]
                if refined_pq_scores:
                    avg_primary_naturalness = sum(s[0] for s in primary_pq_scores) / len(primary_pq_scores)
                    avg_primary_artifacts = sum(s[1] for s in primary_pq_scores) / len(primary_pq_scores)
                    avg_refined_naturalness = sum(s[0] for s in refined_pq_scores) / len(refined_pq_scores)
                    avg_refined_artifacts = sum(s[1] for s in refined_pq_scores) / len(refined_pq_scores)
                    self.logger.info(f"\n[PQ Metric] Evaluation completed!")
                    self.logger.info(f"  Primary - Naturalness: {avg_primary_naturalness:.2f}, Artifacts: {avg_primary_artifacts:.2f}")
                    self.logger.info(f"  Refined - Naturalness: {avg_refined_naturalness:.2f}, Artifacts: {avg_refined_artifacts:.2f}")
                    self.logger.info(f"  Improvement - Naturalness: {avg_refined_naturalness - avg_primary_naturalness:+.2f}, Artifacts: {avg_refined_artifacts - avg_primary_artifacts:+.2f}")
            elif not self.skip_refinement:
                refined_pq_scores = [p.refined_pq_score for p in category_data.data_pairs if p.refined_pq_score is not None]
                if refined_pq_scores:
                    avg_refined_naturalness = sum(s[0] for s in refined_pq_scores) / len(refined_pq_scores)
                    avg_refined_artifacts = sum(s[1] for s in refined_pq_scores) / len(refined_pq_scores)
                    self.logger.info(f"\n[PQ Metric] Evaluation completed!")
                    self.logger.info(f"  Refined - Naturalness: {avg_refined_naturalness:.2f}, Artifacts: {avg_refined_artifacts:.2f}")
        else:
            self.logger.info(f"\n[PQ Metric] Skipped (enable_pq_metric=False)")
            # 将 PQ 相关字段设置为 None
            for pair in category_data.data_pairs:
                if self.enable_primary_scoring:
                    pair.primary_pq_score = None
                    pair.primary_pq_reasoning = None
                pair.refined_pq_score = None
                pair.refined_pq_reasoning = None
        
        # ===== SC指标评价 =====
        if self.enable_sc_metric:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"[SC Metric Evaluation] Starting SC metric evaluation for {category}")
            self.logger.info(f"{'='*80}")
            
            # 确保reward model已加载（如果之前卸载了）
            self._ensure_model_loaded("reward_model")
            
            # SC指标评价 - Primary
            if self.enable_primary_scoring:
                self.logger.info(f"\n[SC Metric] Scoring primary edited images...")
                primary_sc_scores, primary_sc_reasonings, primary_sc_failures = self._score_sc_metric(
                    edited_images=primary_images,
                    edit_instructions=edit_instructions,
                    original_images=original_images,
                    return_failures=True
                )
                for pair, primary_sc_score, primary_sc_reasoning in zip(
                    category_data.data_pairs, primary_sc_scores, primary_sc_reasonings
                ):
                    pair.primary_sc_score = primary_sc_score
                    pair.primary_sc_reasoning = primary_sc_reasoning
            
            # SC指标评价 - Refined（skip_refinement 时跳过）
            if not self.skip_refinement:
                self.logger.info(f"\n[SC Metric] Scoring refined edited images...")
                refined_sc_scores, refined_sc_reasonings, refined_sc_failures = self._score_sc_metric(
                    edited_images=[p.refined_edited_image for p in category_data.data_pairs],
                    edit_instructions=edit_instructions,
                    original_images=original_images,
                    return_failures=True
                )
                for pair, refined_sc_score, refined_sc_reasoning in zip(
                    category_data.data_pairs, refined_sc_scores, refined_sc_reasonings
                ):
                    pair.refined_sc_score = refined_sc_score
                    pair.refined_sc_reasoning = refined_sc_reasoning
            
            # 计算SC指标统计
            if self.skip_refinement and primary_sc_scores:
                avg_primary_editing_success = sum(s[0] for s in primary_sc_scores) / len(primary_sc_scores)
                avg_primary_overediting = sum(s[1] for s in primary_sc_scores) / len(primary_sc_scores)
                self.logger.info(f"\n[SC Metric] Evaluation completed (skip_refinement)! Primary - Editing Success: {avg_primary_editing_success:.2f}, Degree of Overediting: {avg_primary_overediting:.2f}")
            elif self.enable_primary_scoring and primary_sc_scores and refined_sc_scores:
                avg_primary_editing_success = sum(s[0] for s in primary_sc_scores) / len(primary_sc_scores)
                avg_primary_overediting = sum(s[1] for s in primary_sc_scores) / len(primary_sc_scores)
                avg_refined_editing_success = sum(s[0] for s in refined_sc_scores) / len(refined_sc_scores)
                avg_refined_overediting = sum(s[1] for s in refined_sc_scores) / len(refined_sc_scores)
                
                self.logger.info(f"\n[SC Metric] Evaluation completed!")
                self.logger.info(f"  Primary - Editing Success: {avg_primary_editing_success:.2f}, Degree of Overediting: {avg_primary_overediting:.2f}")
                self.logger.info(f"  Refined - Editing Success: {avg_refined_editing_success:.2f}, Degree of Overediting: {avg_refined_overediting:.2f}")
                self.logger.info(f"  Improvement - Editing Success: {avg_refined_editing_success - avg_primary_editing_success:+.2f}, Degree of Overediting: {avg_refined_overediting - avg_primary_overediting:+.2f}")
            elif refined_sc_scores:
                avg_refined_editing_success = sum(s[0] for s in refined_sc_scores) / len(refined_sc_scores)
                avg_refined_overediting = sum(s[1] for s in refined_sc_scores) / len(refined_sc_scores)
                
                self.logger.info(f"\n[SC Metric] Evaluation completed!")
                self.logger.info(f"  Refined - Editing Success: {avg_refined_editing_success:.2f}, Degree of Overediting: {avg_refined_overediting:.2f}")
        else:
            self.logger.info(f"\n[SC Metric] Skipped (enable_sc_metric=False)")
            # 将 SC 相关字段设置为 None
            for pair in category_data.data_pairs:
                if self.enable_primary_scoring:
                    pair.primary_sc_score = None
                    pair.primary_sc_reasoning = None
                pair.refined_sc_score = None
                pair.refined_sc_reasoning = None
        
        # ===== Instruction Following指标评价 =====
        if self.enable_instruction_following_metric:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"[Instruction Following Metric] Starting IF metric evaluation for {category}")
            self.logger.info(f"{'='*80}")
            
            self._ensure_model_loaded("reward_model")
            
            # IF指标评价 - Primary（可选）
            if self.enable_primary_scoring:
                self.logger.info(f"\n[IF Metric] Scoring primary edited images...")
                primary_if_scores, primary_if_reasonings, _ = self._score_instruction_following_metric(
                    edited_images=primary_images,
                    edit_instructions=edit_instructions,
                    original_images=original_images,
                    return_failures=True
                )
                for pair, primary_if_score, primary_if_reasoning in zip(
                    category_data.data_pairs, primary_if_scores, primary_if_reasonings
                ):
                    pair.primary_if_score = primary_if_score
                    pair.primary_if_reasoning = primary_if_reasoning
            
            # IF指标评价 - Refined（skip_refinement 时跳过）
            if not self.skip_refinement:
                self.logger.info(f"\n[IF Metric] Scoring refined edited images...")
                refined_if_scores, refined_if_reasonings, _ = self._score_instruction_following_metric(
                    edited_images=[p.refined_edited_image for p in category_data.data_pairs],
                    edit_instructions=edit_instructions,
                    original_images=original_images,
                    return_failures=True
                )
                for pair, refined_if_score, refined_if_reasoning in zip(
                    category_data.data_pairs, refined_if_scores, refined_if_reasonings
                ):
                    pair.refined_if_score = refined_if_score
                    pair.refined_if_reasoning = refined_if_reasoning
            
            if self.skip_refinement and primary_if_scores:
                avg_primary = sum(primary_if_scores) / len(primary_if_scores)
                self.logger.info(f"\n[IF Metric] Evaluation completed (skip_refinement)! Primary Avg: {avg_primary:.2f}")
            elif self.enable_primary_scoring and primary_if_scores and not self.skip_refinement:
                refined_if_scores = [p.refined_if_score for p in category_data.data_pairs if p.refined_if_score is not None]
                if refined_if_scores:
                    avg_primary = sum(primary_if_scores) / len(primary_if_scores)
                    avg_refined = sum(refined_if_scores) / len(refined_if_scores)
                    self.logger.info(f"\n[IF Metric] Evaluation completed!")
                    self.logger.info(f"  Primary Avg: {avg_primary:.2f}, Refined Avg: {avg_refined:.2f}, Improvement: {avg_refined - avg_primary:+.2f}")
            elif not self.skip_refinement:
                refined_if_scores = [p.refined_if_score for p in category_data.data_pairs if p.refined_if_score is not None]
                if refined_if_scores:
                    avg_refined = sum(refined_if_scores) / len(refined_if_scores)
                    self.logger.info(f"\n[IF Metric] Evaluation completed! Refined Avg: {avg_refined:.2f}")
        else:
            self.logger.info(f"\n[IF Metric] Skipped (enable_instruction_following_metric=False)")
            for pair in category_data.data_pairs:
                if self.enable_primary_scoring:
                    pair.primary_if_score = None
                    pair.primary_if_reasoning = None
                pair.refined_if_score = None
                pair.refined_if_reasoning = None
        
        # 卸载模型
        self._unload_current_model()
    
    def _score_pq_metric(self, 
                         edited_images: List[Image.Image],
                         edit_instructions: List[str],
                         original_images: List[Image.Image],
                         return_failures: bool = False) -> tuple:
        """
        PQ指标评价辅助方法
        
        使用PQ特定的prompt对所有图像进行评价，不依赖类别。
        
        Args:
            edited_images: 编辑后的图像列表
            edit_instructions: 编辑指令列表
            original_images: 原始图像列表
            return_failures: 是否返回失败信息
            
        Returns:
            如果return_failures=False: Tuple[List[List[float]], List[str]]
                (PQ分数列表，每个元素为[naturalness, artifacts], reasoning列表)
            如果return_failures=True: Tuple[List[List[float]], List[str], Dict]
                (PQ分数列表, reasoning列表, 失败信息)
        """
        n = len(edited_images)
        self.logger.info(f"PQ metric scoring {n} images...")
        
        # 获取PQ特定的prompts
        pq_system_prompt = self.prompt_manager.get_pq_system_prompt()
        pq_system_prompts = [pq_system_prompt] * n
        
        pq_user_prompts = [
            self.prompt_manager.get_pq_user_prompt(edit_instruction)
            for edit_instruction in edit_instructions
        ]
        
        # 调用reward model进行评价
        # 注意：PQ评价不需要original_descriptions和rationales
        # batch_score返回的是(scores, reasonings, failures)，其中scores可能是字符串或列表
        result = self.reward_model.batch_score(
            edited_images=edited_images,
            original_descriptions=[""] * n,  # 占位符，PQ不需要
            edit_instructions=edit_instructions,
            system_prompts=pq_system_prompts,
            user_prompts=pq_user_prompts,
            original_images=original_images,
            rationales=None,  # PQ不需要rationale
            return_failures=True
        )
        
        # 处理返回结果：根据返回类型判断
        if len(result) == 3:
            pq_scores, pq_reasonings, pq_failures = result
        else:
            # 向后兼容：如果没有failures，创建空的failures字典
            pq_scores, pq_reasonings = result
            pq_failures = {
                'failed_gpus': [],
                'failed_sample_indices': [],
                'failed_sample_count': 0,
                'total_samples': n,
                'error_messages': {}
            }
        
        # 处理返回结果：pq_scores可能是列表格式[naturalness, artifacts]或字符串"yes"/"no"
        # 如果是字符串，转换为默认值
        processed_scores = []
        for i, score in enumerate(pq_scores):
            if isinstance(score, list) and len(score) >= 2:
                # 已经是列表格式
                try:
                    processed_scores.append([float(score[0]), float(score[1])])
                except (ValueError, TypeError, IndexError) as e:
                    self.logger.warning(f"PQ score format error at index {i}: {score}, error: {e}, using default [5.0, 5.0]")
                    processed_scores.append([5.0, 5.0])
            else:
                # 如果不是列表格式，使用默认值（可能是解析失败）
                self.logger.warning(f"PQ score is not in list format at index {i}: {score} (type: {type(score)}), using default [5.0, 5.0]")
                processed_scores.append([5.0, 5.0])
        
        self.logger.info(f"PQ metric scoring completed for {n} images")
        
        if return_failures:
            return processed_scores, pq_reasonings, pq_failures
        else:
            return processed_scores, pq_reasonings
    
    def _score_sc_metric(self, 
                         edited_images: List[Image.Image],
                         edit_instructions: List[str],
                         original_images: List[Image.Image],
                         return_failures: bool = False) -> tuple:
        """
        SC指标评价辅助方法
        
        使用SC特定的prompt对所有图像进行评价，不依赖类别。
        
        Args:
            edited_images: 编辑后的图像列表
            edit_instructions: 编辑指令列表
            original_images: 原始图像列表
            return_failures: 是否返回失败信息
            
        Returns:
            如果return_failures=False: Tuple[List[List[float]], List[str]]
                (SC分数列表，每个元素为[editing_success, degree_of_overediting], reasoning列表)
            如果return_failures=True: Tuple[List[List[float]], List[str], Dict]
                (SC分数列表, reasoning列表, 失败信息)
        """
        n = len(edited_images)
        self.logger.info(f"SC metric scoring {n} images...")
        
        # 获取SC特定的prompts
        sc_system_prompt = self.prompt_manager.get_sc_system_prompt()
        sc_system_prompts = [sc_system_prompt] * n
        
        sc_user_prompts = [
            self.prompt_manager.get_sc_user_prompt(edit_instruction)
            for edit_instruction in edit_instructions
        ]
        
        # 调用reward model进行评价
        # 注意：SC评价不需要original_descriptions和rationales
        result = self.reward_model.batch_score(
            edited_images=edited_images,
            original_descriptions=[""] * n,  # 占位符，SC不需要
            edit_instructions=edit_instructions,
            system_prompts=sc_system_prompts,
            user_prompts=sc_user_prompts,
            original_images=original_images,
            rationales=None,  # SC不需要rationale
            return_failures=True
        )
        
        # 处理返回结果：根据返回类型判断
        if len(result) == 3:
            sc_scores, sc_reasonings, sc_failures = result
        else:
            # 向后兼容：如果没有failures，创建空的failures字典
            sc_scores, sc_reasonings = result
            sc_failures = {
                'failed_gpus': [],
                'failed_sample_indices': [],
                'failed_sample_count': 0,
                'total_samples': n,
                'error_messages': {}
            }
        
        # 处理返回结果：sc_scores可能是列表格式[editing_success, degree_of_overediting]或字符串"yes"/"no"
        # 如果是字符串，转换为默认值
        processed_scores = []
        for i, score in enumerate(sc_scores):
            if isinstance(score, list) and len(score) >= 2:
                # 已经是列表格式
                try:
                    processed_scores.append([float(score[0]), float(score[1])])
                except (ValueError, TypeError, IndexError) as e:
                    self.logger.warning(f"SC score format error at index {i}: {score}, error: {e}, using default [5.0, 5.0]")
                    processed_scores.append([5.0, 5.0])
            else:
                # 如果不是列表格式，使用默认值（可能是解析失败）
                self.logger.warning(f"SC score is not in list format at index {i}: {score} (type: {type(score)}), using default [5.0, 5.0]")
                processed_scores.append([5.0, 5.0])
        
        self.logger.info(f"SC metric scoring completed for {n} images")
        
        if return_failures:
            return processed_scores, sc_reasonings, sc_failures
        else:
            return processed_scores, sc_reasonings
    
    def _score_instruction_following_metric(self, 
                                            edited_images: List[Image.Image],
                                            edit_instructions: List[str],
                                            original_images: List[Image.Image],
                                            return_failures: bool = False) -> tuple:
        """
        Instruction Following指标评价辅助方法
        
        评估编辑结果对指令的遵循程度，输出单分0-10。
        
        Returns:
            如果return_failures=False: Tuple[List[float], List[str]]
            如果return_failures=True: Tuple[List[float], List[str], Dict]
        """
        n = len(edited_images)
        self.logger.info(f"Instruction Following metric scoring {n} images...")
        
        if_system_prompt = self.prompt_manager.get_instruction_following_system_prompt()
        if_system_prompts = [if_system_prompt] * n
        if_user_prompts = [
            self.prompt_manager.get_instruction_following_user_prompt(edit_instruction)
            for edit_instruction in edit_instructions
        ]
        
        result = self.reward_model.batch_score(
            edited_images=edited_images,
            original_descriptions=[""] * n,
            edit_instructions=edit_instructions,
            system_prompts=if_system_prompts,
            user_prompts=if_user_prompts,
            original_images=original_images,
            rationales=None,
            return_failures=True
        )
        
        if len(result) == 3:
            if_scores, if_reasonings, if_failures = result
        else:
            if_scores, if_reasonings = result
            if_failures = {
                'failed_gpus': [], 'failed_sample_indices': [], 'failed_sample_count': 0,
                'total_samples': n, 'error_messages': {}
            }
        
        # 处理返回：IF为单分0-10，可能为int/float或错误解析为list
        processed_scores = []
        for i, score in enumerate(if_scores):
            if isinstance(score, (int, float)):
                processed_scores.append(float(score))
            elif isinstance(score, list) and len(score) >= 1:
                try:
                    processed_scores.append(float(score[0]))
                except (ValueError, TypeError):
                    self.logger.warning(f"IF score format error at index {i}: {score}, using default 5.0")
                    processed_scores.append(5.0)
            else:
                self.logger.warning(f"IF score invalid at index {i}: {score} (type: {type(score)}), using default 5.0")
                processed_scores.append(5.0)
        
        self.logger.info(f"Instruction Following metric scoring completed for {n} images")
        
        if return_failures:
            return processed_scores, if_reasonings, if_failures
        return processed_scores, if_reasonings
    
    def _stage5_statistics(self, category: str, category_data: IterativeCategoryData):
        """
        统计记录
        
        计算类别统计信息（根据enable_primary_scoring决定是否包含primary统计）
        """
        category_data.calculate_statistics()
        
        stats = category_data.statistics
        self.logger.info(f"\nCategory: {category}")
        
        if self.skip_refinement:
            # skip_refinement 模式：仅输出 primary 统计
            self.logger.info(f"  Primary Yes Rate: {stats.get('primary_yes_rate', 0.0):.2f}% ({stats.get('primary_yes_count', 0)}/{stats.get('total_count', 0)})")
        elif self.enable_primary_scoring:
            # Primary + Refined 统计
            self.logger.info(f"  Primary Yes Rate: {stats.get('primary_yes_rate', 0.0):.2f}% ({stats.get('primary_yes_count', 0)}/{stats.get('total_count', 0)})")
            self.logger.info(f"  Refined Yes Rate: {stats.get('refined_yes_rate', 0.0):.2f}% ({stats.get('refined_yes_count', 0)}/{stats.get('total_count', 0)})")
            self.logger.info(f"  Improvement Rate: {stats.get('improvement_rate', 0.0):.2f}% ({stats.get('improved_count', 0)} improved)")
            self.logger.info(f"  Maintained Rate: {stats.get('maintained_rate', 0.0):.2f}% ({stats.get('maintained_count', 0)} maintained)")
            self.logger.info(f"  Regression Rate: {stats.get('regression_rate', 0.0):.2f}% ({stats.get('regression_count', 0)} regressed)")
            self.logger.info(f"  Unchanged Rate: {stats.get('unchanged_rate', 0.0):.2f}% ({stats.get('unchanged_count', 0)} unchanged)")
        else:
            self.logger.info(f"  Refined Yes Rate: {stats.get('refined_yes_rate', 0.0):.2f}% ({stats.get('refined_yes_count', 0)}/{stats.get('total_count', 0)})")
            self.logger.info(f"  Refined Avg (Yes Rate): {stats.get('refined_avg', 0.0):.2f}%")
    
    def _get_gpu_info(self, refinement_model) -> tuple:
        """
        获取refinement_model的GPU设备信息
        
        Args:
            refinement_model: refinement diffusion模型实例
            
        Returns:
            (device_ids, num_gpus) 元组
        """
        # 尝试从模型获取device_ids
        if hasattr(refinement_model, 'device_ids'):
            device_ids = refinement_model.device_ids
            num_gpus = len(device_ids)
        elif hasattr(refinement_model, 'num_gpus'):
            num_gpus = refinement_model.num_gpus
            if hasattr(refinement_model, 'config') and refinement_model.config:
                device_ids = refinement_model.config.get("device_ids", list(range(num_gpus)))
            else:
                device_ids = list(range(num_gpus))
        else:
            # 从配置中读取
            refinement_config = self.config["diffusion_model"].get("refinement") or self.config["diffusion_model"].get("primary")
            if refinement_config and "params" in refinement_config:
                device_ids = refinement_config["params"].get("device_ids", [0])
                num_gpus = len(device_ids) if isinstance(device_ids, list) else 1
                if not isinstance(device_ids, list):
                    device_ids = [device_ids]
            else:
                # 默认使用单个GPU
                device_ids = [0]
                num_gpus = 1
        
        return device_ids, num_gpus
    
    def _assign_pairs_to_gpus(self, pairs: List[IterativeDataPair], device_ids: List[int], num_gpus: int) -> Dict[int, List[IterativeDataPair]]:
        """
        将pairs随机分配到不同的GPU
        
        Args:
            pairs: 数据对列表
            device_ids: GPU设备ID列表
            num_gpus: GPU数量
            
        Returns:
            {gpu_id: [list of pairs]} 字典
        """
        # 创建pairs的副本并随机打乱
        pairs_copy = list(pairs)
        random.shuffle(pairs_copy)
        
        # 按round-robin方式分配到各GPU
        gpu_tasks = {gpu_id: [] for gpu_id in device_ids}
        for idx, pair in enumerate(pairs_copy):
            gpu_idx = idx % num_gpus
            gpu_id = device_ids[gpu_idx]
            gpu_tasks[gpu_id].append(pair)
        
        # 记录分配情况
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"[Task Assignment] Randomly assigned {len(pairs)} pairs to {num_gpus} GPUs:")
        self.logger.info(f"{'='*80}")
        for gpu_id in device_ids:
            num_tasks = len(gpu_tasks[gpu_id])
            self.logger.info(f"  GPU {gpu_id}: {num_tasks} pairs")
        self.logger.info(f"{'='*80}\n")
        
        return gpu_tasks
    
    def _process_pairs_on_gpu(self, refinement_model, gpu_id: int, pairs: List[IterativeDataPair], category: str) -> List[Dict]:
        """
        在指定GPU上处理分配给它的pairs，进行串行迭代refinement
        
        Args:
            refinement_model: refinement diffusion模型实例
            gpu_id: GPU设备ID
            pairs: 分配给该GPU的pairs列表
            category: 类别名称
            
        Returns:
            处理结果列表，每个元素为 {"pair": pair, "refined_image": Image, "success": bool}
        """
        results = []
        
        self.logger.info(f"[GPU {gpu_id}] Starting processing {len(pairs)} pairs...")
        
        for pair_idx, pair in enumerate(pairs, 1):
            # 获取该pair的指令列表
            if pair.re_edit_instructions and len(pair.re_edit_instructions) > 0:
                instructions_list = pair.re_edit_instructions
            else:
                # 如果没有独立的指令列表，fallback到使用拼接后的指令（包装为列表）
                instructions_list = [pair.re_edit_instruction] if pair.re_edit_instruction else []
            
            num_iterations = len(instructions_list)
            pair.refinement_iterations = num_iterations
            
            if num_iterations == 0:
                self.logger.warning(f"[GPU {gpu_id}] Pair {pair.pair_id}: No re-edit instructions found, using primary image as refined image")
                results.append({
                    "pair": pair,
                    "refined_image": pair.primary_edited_image,
                    "success": True
                })
                continue
            
            # 记录迭代信息（同时输出到日志和控制台）
            iteration_info = f"[GPU {gpu_id}] [Pair {pair.pair_id} ({pair_idx}/{len(pairs)})] Starting iterative refinement with {num_iterations} iteration(s)"
            self.logger.info(iteration_info)
            print(f"🔄 {iteration_info}")
            
            # 打印迭代指令详情
            if instructions_list:
                self.logger.info(f"[GPU {gpu_id}]   Iteration instructions for {pair.pair_id}:")
                print(f"📋 [GPU {gpu_id}] Iteration instructions for {pair.pair_id}:")
                for idx, instr in enumerate(instructions_list, 1):
                    instr_msg = f"[GPU {gpu_id}]     [{idx}/{num_iterations}] {instr}"
                    self.logger.info(instr_msg)
                    print(f"   {instr_msg}")
            
            # 初始化：第一轮使用 primary_edited_image 作为输入
            current_image = pair.primary_edited_image
            success = True
            
            # 迭代refinement：每次使用一个指令，结果作为下一轮的输入
            for iter_idx, instruction in enumerate(instructions_list, 1):
                iteration_msg = f"[GPU {gpu_id}]   [Iteration {iter_idx}/{num_iterations}] Pair {pair.pair_id}: {instruction}"
                self.logger.info(iteration_msg)
                print(f"⚙️  {iteration_msg}")
                
                try:
                    # 执行单次refinement（与现有diffusion方式适配）
                    # 在迭代refinement模式下，直接使用指定GPU，避免batch_sync导致的同步等待
                    if hasattr(refinement_model, 'edit_image'):
                        # 如果模型支持target_gpu_id参数，直接指定GPU（避免batch_sync）
                        # 否则使用默认行为（可能触发batch_sync，但至少能工作）
                        if hasattr(refinement_model, 'device_ids') and gpu_id in refinement_model.device_ids:
                            # 直接指定目标GPU，绕过batch_sync机制
                            refined_image = refinement_model.edit_image(
                                current_image, 
                                instruction,
                                target_gpu_id=gpu_id  # 指定GPU，避免轮询分配和同步等待
                            )
                        else:
                            # Fallback：使用默认行为（可能不是最优，但至少能工作）
                            refined_image = refinement_model.edit_image(
                                current_image, 
                                instruction,
                                enable_batch_sync=False  # 禁用同步，让各GPU独立处理
                            )
                    elif hasattr(refinement_model, 'batch_edit'):
                        # 如果模型不支持单图编辑，使用batch_edit（传入单个元素）
                        refined_results = refinement_model.batch_edit(
                            images=[current_image],
                            instructions=[instruction],
                            enable_batch_sync=False  # 禁用同步，让各GPU独立处理
                        )
                        refined_image = refined_results[0] if refined_results else current_image
                    else:
                        raise RuntimeError(f"Refinement model does not support edit_image or batch_edit")
                    
                    # 关键：将本次refinement的结果作为下一轮的输入
                    current_image = refined_image
                    
                    # 保存迭代中间结果（过程结果）
                    if self.save_images:
                        self._save_iteration_image(
                            refined_image, 
                            category, 
                            pair.pair_id, 
                            iter_idx, 
                            num_iterations
                        )
                    
                except Exception as e:
                    self.logger.error(f"[GPU {gpu_id}]   ✗ Iteration {iter_idx} failed for pair {pair.pair_id}: {e}")
                    self.logger.warning(f"[GPU {gpu_id}]   Using previous iteration result as fallback")
                    success = False
                    # 如果某次迭代失败，使用上一轮的结果（或primary image）
                    if iter_idx == 1:
                        current_image = pair.primary_edited_image
                    # 否则 current_image 保持为上一轮的结果
            
            # 保存最终结果（最后一轮refinement的结果）
            pair.refined_edited_image = current_image
            results.append({
                "pair": pair,
                "refined_image": current_image,
                "success": success
            })
            
            completion_msg = f"[GPU {gpu_id}] [Pair {pair.pair_id}] Iterative refinement completed: {num_iterations} iteration(s)"
            self.logger.info(completion_msg)
            print(f"✅ {completion_msg}")
        
        summary_msg = f"[GPU {gpu_id}] ✅ Completed processing {len(pairs)} pairs"
        self.logger.info(summary_msg)
        print(f"🎉 {summary_msg}")
        return results
    
    def _parallel_iterative_refinement(self, refinement_model, gpu_tasks: Dict[int, List[IterativeDataPair]], 
                                       device_ids: List[int], category: str, category_data: IterativeCategoryData) -> List[Image.Image]:
        """
        并行执行迭代refinement，等待所有GPU完成
        
        Args:
            refinement_model: refinement diffusion模型实例
            gpu_tasks: {gpu_id: [list of pairs]} 任务分配字典
            device_ids: GPU设备ID列表
            category: 类别名称
            category_data: 类别数据（用于保持原始顺序）
            
        Returns:
            按原始顺序排列的refined images列表
        """
        num_gpus = len(device_ids)
        refined_images = [None] * len(category_data.data_pairs)
        
        # 创建pair_id到索引的映射（用于保持原始顺序）
        pair_id_to_index = {pair.pair_id: idx for idx, pair in enumerate(category_data.data_pairs)}
        
        # 使用ThreadPoolExecutor并行执行
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for gpu_id in device_ids:
                if gpu_tasks[gpu_id]:  # 只提交有任务的GPU
                    future = executor.submit(
                        self._process_pairs_on_gpu,
                        refinement_model,
                        gpu_id,
                        gpu_tasks[gpu_id],
                        category
                    )
                    futures[future] = gpu_id
            
            # 等待所有GPU完成并收集结果
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"[Synchronization] Waiting for all {len(futures)} GPUs to complete...")
            self.logger.info(f"{'='*80}")
            
            completed_count = 0
            for future in as_completed(futures):
                gpu_id = futures[future]
                completed_count += 1
                try:
                    gpu_results = future.result()
                    self.logger.info(f"[GPU {gpu_id}] ✅ Completed ({completed_count}/{len(futures)} GPUs done)")
                    
                    # 将结果按原始顺序组装
                    for result in gpu_results:
                        pair = result["pair"]
                        refined_image = result["refined_image"]
                        pair_idx = pair_id_to_index[pair.pair_id]
                        refined_images[pair_idx] = refined_image
                        
                        if not result["success"]:
                            self.logger.warning(f"[GPU {gpu_id}] Pair {pair.pair_id} had some failed iterations, using fallback result")
                    
                except Exception as e:
                    self.logger.error(f"[GPU {gpu_id}] ❌ Failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # 对于失败的GPU，使用primary image作为fallback
                    for pair in gpu_tasks[gpu_id]:
                        pair_idx = pair_id_to_index[pair.pair_id]
                        if refined_images[pair_idx] is None:
                            refined_images[pair_idx] = pair.primary_edited_image
                            pair.refined_edited_image = pair.primary_edited_image
                            self.logger.warning(f"[GPU {gpu_id}] Using primary image as fallback for pair {pair.pair_id}")
            
            self.logger.info(f"{'='*80}")
            self.logger.info(f"[Synchronization] All {len(futures)} GPUs completed!")
            self.logger.info(f"{'='*80}\n")
        
        # 验证所有结果都已收集
        missing_count = sum(1 for img in refined_images if img is None)
        if missing_count > 0:
            self.logger.warning(f"Warning: {missing_count} pairs have None refined images, using primary images as fallback")
            for idx, img in enumerate(refined_images):
                if img is None:
                    pair = category_data.data_pairs[idx]
                    refined_images[idx] = pair.primary_edited_image
                    pair.refined_edited_image = pair.primary_edited_image
        
        return refined_images
    
    def _get_category_dir(self, category: str) -> str:
        """
        获取 category 对应的目录名（用于图片保存/加载）
        支持 category_to_dir 映射以兼容中文目录名（如从 primary_images_dir 加载时）
        """
        if self.category_to_dir and category in self.category_to_dir:
            return self.category_to_dir[category]
        return category
    
    # 将 pair_id/文件名中可能包含的中文 subset 替换为英文（用于保存时生成统一英文学名）
    _SUBSET_ZH_TO_EN = {"物理": "physical", "环境": "environmental", "社会": "cultural", "因果": "causal", "指代": "referential"}
    
    def _sanitize_filename_part(self, s: str) -> str:
        """将 pair_id 中的中文 subset 替换为英文，避免保存的文件名包含中文"""
        if not s:
            return s
        result = s
        for zh, en in self._SUBSET_ZH_TO_EN.items():
            result = result.replace(zh, en)
        return result
    
    def _save_image(self, image: Image.Image, category: str, pair_id: str, stage: str):
        """
        保存图像
        
        Args:
            image: 图像对象
            category: 类别名称（来自 config.data.categories，英文 subset 时为 physical/environmental 等）
            pair_id: 数据对ID
            stage: 阶段标识（primary/refined）
        """
        dir_name = self._get_category_dir(category)
        category_dir = os.path.join(self.output_dir, dir_name)
        os.makedirs(category_dir, exist_ok=True)
        
        # 将 pair_id 中的中文 subset 替换为英文，避免保存的文件名包含中文（如 00008_物理_medium -> 00008_physical_medium）
        safe_pair_id = self._sanitize_filename_part(pair_id)
        
        # 如果配置了 primary_images_dir 且保存的是 refined 图片，在文件名中添加标识以避免覆盖
        # 如果使用了 use_original_as_primary，也要在文件名中添加标识
        # 注意：如果 use_original_as_primary=True，不应该包含 primary_images_dir 的标识（因为实际未使用）
        if stage == "refined":
            parts = []
            if self.use_original_as_primary:
                parts.append("use_original_as_primary")
                # 如果使用原图作为 primary，不应该包含 primary_images_dir 的标识
            elif self.primary_images_dir:
                # 只有在不使用原图作为 primary 时，才添加 primary_images_dir 标识
                primary_dir_name = os.path.basename(os.path.normpath(self.primary_images_dir))
                parts.append(f"refined_only_{primary_dir_name}_under_qwen_image_edit_small_data_sft_mllm_ablation")
            if self.enable_iterative_refinement:
                parts.append("iterative_refinement")
            
            if parts:
                filename = f"{safe_pair_id}_{'_'.join(parts)}.png"
            else:
                filename = f"{safe_pair_id}_{stage}.png"
        else:
            filename = f"{safe_pair_id}_{stage}.png"
        
        filepath = os.path.join(category_dir, filename)
        image.save(filepath)
    
    def _save_iteration_image(self, image: Image.Image, category: str, pair_id: str, 
                              iteration_idx: int, total_iterations: int):
        """
        保存迭代refinement的中间结果（过程结果）
        
        Args:
            image: 图像对象
            category: 类别名称（来自 config.data.categories，英文 subset 时为 physical/environmental 等）
            pair_id: 数据对ID
            iteration_idx: 当前迭代编号（从1开始）
            total_iterations: 总迭代次数
        """
        dir_name = self._get_category_dir(category)
        category_dir = os.path.join(self.output_dir, dir_name)
        os.makedirs(category_dir, exist_ok=True)
        
        # 创建迭代结果子目录
        if self.use_original_as_primary:
            iterations_dir = os.path.join(category_dir, "iterations_w_original_as_primary")
        else:
            iterations_dir = os.path.join(category_dir, "iterations_wo_original_as_primary")
        os.makedirs(iterations_dir, exist_ok=True)
        
        # 将 pair_id 中的中文 subset 替换为英文
        safe_pair_id = self._sanitize_filename_part(pair_id)
        
        # 命名规则：明确标识为过程结果
        # 格式：{pair_id}_refined_iteration_{iter_idx}_of_{total_iterations}.png
        filename = f"{safe_pair_id}_refined_iteration_{iteration_idx}_of_{total_iterations}.png"
        filepath = os.path.join(iterations_dir, filename)
        image.save(filepath)
        
        # 记录保存信息（过程结果）
        save_msg = f"💾 Saved iteration {iteration_idx}/{total_iterations} intermediate result for pair {pair_id}: {filename}"
        self.logger.info(save_msg)
        print(save_msg)
    
    def _use_original_as_primary(self, category: str, category_data: IterativeCategoryData):
        """
        直接使用原图作为 primary edited images（trick）
        
        Args:
            category: 类别名称
            category_data: 类别数据
        """
        self.logger.info(f"Using original images as primary images for category: {category}")
        
        for pair in category_data.data_pairs:
            # 确保 original_image 已解码
            if pair.original_image is None:
                if pair.original_image_b64:
                    pair.original_image = decode_base64_image(pair.original_image_b64)
                else:
                    raise RuntimeError(
                        f"original_image and original_image_b64 are both None for pair {pair.pair_id} "
                        f"in category {category}. Cannot use original as primary."
                    )
            
            # 直接使用原图作为 primary_edited_image
            pair.primary_edited_image = pair.original_image
        
        self.logger.info(f"Set {len(category_data.data_pairs)} pairs to use original images as primary images")
    
    def _load_primary_images_from_dir(self, category: str, category_data: IterativeCategoryData):
        """
        从指定目录加载 primary edited images
        
        文件命名格式：{pair_id}{primary_image_suffix}
        目录结构：{primary_images_dir}/{category}/{pair_id}{primary_image_suffix}
        
        Args:
            category: 类别名称
            category_data: 类别数据
        """
        if not self.primary_images_dir:
            raise ValueError("primary_images_dir not configured. Cannot load primary images from directory.")
        
        dir_name = self._get_category_dir(category)
        category_dir = os.path.join(self.primary_images_dir, dir_name)
        
        if not os.path.exists(category_dir):
            raise FileNotFoundError(f"Primary images directory not found: {category_dir}")
        
        self.logger.info(f"Loading primary edited images from: {category_dir}")
        self.logger.info(f"Using primary image suffix: {self.primary_image_suffix}")
        
        loaded_count = 0
        missing_count = 0
        
        for pair in category_data.data_pairs:
            # 构建文件路径：{pair_id}{primary_image_suffix}
            # 尝试两种文件名：原始 pair_id（可能含中文）和 sanitized pair_id（英文），以兼容新旧文件
            filename_orig = f"{pair.pair_id}{self.primary_image_suffix}"
            filename_safe = f"{self._sanitize_filename_part(pair.pair_id)}{self.primary_image_suffix}"
            filepath = os.path.join(category_dir, filename_orig)
            if not os.path.exists(filepath):
                filepath = os.path.join(category_dir, filename_safe)
            
            if os.path.exists(filepath):
                try:
                    # 加载图像
                    pair.primary_edited_image = Image.open(filepath).convert('RGB')
                    loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to load image {filepath}: {e}. Using original image as fallback.")
                    # 使用原图作为 fallback
                    if pair.original_image is None:
                        pair.original_image = decode_base64_image(pair.original_image_b64)
                    pair.primary_edited_image = pair.original_image
                    missing_count += 1
            else:
                self.logger.warning(f"Primary image not found: {filepath}. Using original image as fallback.")
                # 使用原图作为 fallback
                if pair.original_image is None:
                    pair.original_image = decode_base64_image(pair.original_image_b64)
                pair.primary_edited_image = pair.original_image
                missing_count += 1
        
        self.logger.info(f"Loaded {loaded_count}/{len(category_data.data_pairs)} primary images from directory")
        if missing_count > 0:
            self.logger.warning(f"Missing {missing_count} primary images, used original images as fallback")
    
    def _generate_report(self, benchmark_data: IterativeBenchmarkData) -> Dict[str, Any]:
        """
        生成对比报告
        
        Args:
            benchmark_data: Benchmark数据
        
        Returns:
            报告字典（与standard pipeline兼容）
        """
        global_stats = benchmark_data.calculate_global_statistics()
        
        # 如果跳过了 Stage4，不输出评价相关的统计信息
        if not self.skip_stage4:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("GLOBAL STATISTICS")
            self.logger.info("=" * 80)
            self.logger.info(f"Total Pairs: {global_stats['total_pairs']}")
            if self.skip_refinement:
                self.logger.info(f"Primary Avg Score: {global_stats['global_primary_avg']:.3f}")
            else:
                self.logger.info(f"Refined Avg Score: {global_stats['global_refined_avg']:.3f}")
            
            if self.enable_pq_metric:
                self.logger.info(f"\n{'='*80}")
                self.logger.info("PQ METRIC STATISTICS (using min score)")
                self.logger.info(f"{'='*80}")
                if self.skip_refinement:
                    self.logger.info(f"Primary PQ Avg: {global_stats.get('global_primary_pq_avg', 0.0):.2f}")
                else:
                    self.logger.info(f"Refined PQ Avg: {global_stats.get('global_refined_pq_avg', 0.0):.2f}")
            
            if self.enable_sc_metric:
                self.logger.info(f"\n{'='*80}")
                self.logger.info("SC METRIC STATISTICS (using min score)")
                self.logger.info(f"{'='*80}")
                if self.skip_refinement:
                    self.logger.info(f"Primary SC Avg: {global_stats.get('global_primary_sc_avg', 0.0):.2f}")
                else:
                    self.logger.info(f"Refined SC Avg: {global_stats.get('global_refined_sc_avg', 0.0):.2f}")
            
            if self.enable_instruction_following_metric:
                self.logger.info(f"\n{'='*80}")
                self.logger.info("INSTRUCTION FOLLOWING METRIC STATISTICS")
                self.logger.info(f"{'='*80}")
                if self.skip_refinement:
                    self.logger.info(f"Primary IF Avg: {global_stats.get('global_primary_if_avg', 0.0):.2f}")
                else:
                    self.logger.info(f"Refined IF Avg: {global_stats.get('global_refined_if_avg', 0.0):.2f}")
        else:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("GLOBAL SUMMARY (Stage 4 Scoring Skipped)")
            self.logger.info("=" * 80)
            self.logger.info(f"Total Pairs: {global_stats['total_pairs']}")
            self.logger.info("Note: Scoring statistics are not available (Stage 4 was skipped)")
        
        # 构建报告数据结构
        category_statistics = {}
        mllm_analysis_details = {}  # 新增：存储MLLM分析详情
        
        for category_name, category_data in benchmark_data.categories.items():
            stats = category_data.statistics
            
            # 如果跳过了 Stage4，不包含评价相关的统计信息
            if not self.skip_stage4:
                if self.skip_refinement:
                    # skip_refinement 模式：仅使用 primary_* 作为主统计
                    category_statistics[category_name] = {
                        "primary_yes_count": stats.get("primary_yes_count", 0),
                        "primary_no_count": stats.get("primary_no_count", 0),
                        "primary_yes_rate": stats.get("primary_yes_rate", 0.0),
                        "primary_avg": stats.get("primary_avg", 0.0),
                        "total_count": stats.get("total_count", 0),
                        "average": stats.get("primary_avg", 0.0),
                        "std": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "count": stats.get("total_count", 0)
                    }
                else:
                    category_statistics[category_name] = {
                        "refined_yes_count": stats.get("refined_yes_count", 0),
                        "refined_no_count": stats.get("refined_no_count", 0),
                        "refined_yes_rate": stats.get("refined_yes_rate", 0.0),
                        "refined_avg": stats.get("refined_avg", 0.0),
                        "total_count": stats.get("total_count", 0),
                        "average": stats.get("refined_avg", 0.0),
                        "std": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "count": stats.get("total_count", 0)
                    }
                
                # Primary统计（非 skip_refinement 时，enable_primary_scoring 时包含）
                if self.enable_primary_scoring and not self.skip_refinement:
                    category_statistics[category_name].update({
                        "primary_yes_count": stats.get("primary_yes_count", 0),
                        "primary_no_count": stats.get("primary_no_count", 0),
                        "primary_yes_rate": stats.get("primary_yes_rate", 0.0),
                        "primary_avg": stats.get("primary_avg", 0.0),
                        # 改进统计
                        "improved_count": stats.get("improved_count", 0),
                        "improvement_rate": stats.get("improvement_rate", 0.0),
                        "maintained_count": stats.get("maintained_count", 0),
                        "maintained_rate": stats.get("maintained_rate", 0.0),
                        "regression_count": stats.get("regression_count", 0),
                        "regression_rate": stats.get("regression_rate", 0.0),
                        "unchanged_count": stats.get("unchanged_count", 0),
                        "unchanged_rate": stats.get("unchanged_rate", 0.0)
                    })
                
                # PQ指标统计（只在启用时包含）
                if self.enable_pq_metric:
                    if self.skip_refinement:
                        pq_stats = {
                            "primary_pq_avg": stats.get("primary_pq_avg", 0.0),
                            "primary_pq_avg_naturalness": stats.get("primary_pq_avg_naturalness", 0.0),
                            "primary_pq_avg_artifacts": stats.get("primary_pq_avg_artifacts", 0.0)
                        }
                    else:
                        pq_stats = {
                            "refined_pq_avg": stats.get("refined_pq_avg", 0.0),
                            "refined_pq_avg_naturalness": stats.get("refined_pq_avg_naturalness", 0.0),
                            "refined_pq_avg_artifacts": stats.get("refined_pq_avg_artifacts", 0.0)
                        }
                    if self.enable_primary_scoring and not self.skip_refinement:
                        pq_stats.update({
                            "primary_pq_avg": stats.get("primary_pq_avg", 0.0),
                            "primary_pq_avg_naturalness": stats.get("primary_pq_avg_naturalness", 0.0),
                            "primary_pq_avg_artifacts": stats.get("primary_pq_avg_artifacts", 0.0),
                            "pq_improvement": stats.get("pq_improvement", 0.0),
                            "pq_improvement_naturalness": stats.get("pq_improvement_naturalness", 0.0),
                            "pq_improvement_artifacts": stats.get("pq_improvement_artifacts", 0.0)
                        })
                    category_statistics[category_name].update(pq_stats)
                
                # SC指标统计（只在启用时包含）
                if self.enable_sc_metric:
                    if self.skip_refinement:
                        sc_stats = {
                            "primary_sc_avg": stats.get("primary_sc_avg", 0.0),
                            "primary_sc_avg_editing_success": stats.get("primary_sc_avg_editing_success", 0.0),
                            "primary_sc_avg_overediting": stats.get("primary_sc_avg_overediting", 0.0)
                        }
                    else:
                        sc_stats = {
                            "refined_sc_avg": stats.get("refined_sc_avg", 0.0),
                            "refined_sc_avg_editing_success": stats.get("refined_sc_avg_editing_success", 0.0),
                            "refined_sc_avg_overediting": stats.get("refined_sc_avg_overediting", 0.0)
                        }
                    if self.enable_primary_scoring and not self.skip_refinement:
                        sc_stats.update({
                            "primary_sc_avg": stats.get("primary_sc_avg", 0.0),
                            "primary_sc_avg_editing_success": stats.get("primary_sc_avg_editing_success", 0.0),
                            "primary_sc_avg_overediting": stats.get("primary_sc_avg_overediting", 0.0),
                            "sc_improvement": stats.get("sc_improvement", 0.0),
                            "sc_improvement_editing_success": stats.get("sc_improvement_editing_success", 0.0),
                            "sc_improvement_overediting": stats.get("sc_improvement_overediting", 0.0)
                        })
                    category_statistics[category_name].update(sc_stats)
                
                # IF指标统计（只在启用时包含）
                if self.enable_instruction_following_metric:
                    if self.skip_refinement:
                        if_stats = {"primary_if_avg": stats.get("primary_if_avg", 0.0)}
                    else:
                        if_stats = {"refined_if_avg": stats.get("refined_if_avg", 0.0)}
                    if self.enable_primary_scoring and not self.skip_refinement:
                        if_stats.update({
                            "primary_if_avg": stats.get("primary_if_avg", 0.0),
                            "if_improvement": stats.get("if_improvement", 0.0)
                        })
                    category_statistics[category_name].update(if_stats)
            else:
                # 如果跳过 Stage4，只包含基础统计信息（不包含评价相关）
                category_statistics[category_name] = {
                    "total_count": stats.get("total_count", 0),
                    "note": "Scoring statistics not available (Stage 4 was skipped)"
                }
            
            # 收集该类别的MLLM分析详情
            mllm_analysis_details[category_name] = []
            for pair in category_data.data_pairs:
                sample_detail = {
                    "pair_id": pair.pair_id,
                    "original_instruction": pair.edit_instruction,
                    "cot_reasoning": pair.cot_reasoning,
                    "re_edit_instruction": pair.re_edit_instruction,  # 拼接后的字符串（向后兼容）
                    "re_edit_instructions": pair.re_edit_instructions,  # 独立的指令列表（用于迭代refinement）
                    "mllm_raw_output": pair.mllm_raw_output,
                    "refinement_iterations": pair.refinement_iterations,  # 记录refinement迭代次数
                }
                
                # 如果未跳过 Stage4，包含评价结果
                if not self.skip_stage4:
                    if self.skip_refinement:
                        # skip_refinement 模式：仅包含 primary_* 评价结果
                        sample_detail.update({
                            "primary_score": pair.primary_score,
                            "primary_score_reasoning": pair.primary_score_reasoning
                        })
                        if self.enable_pq_metric:
                            sample_detail.update({
                                "primary_pq_score": pair.primary_pq_score,
                                "primary_pq_reasoning": pair.primary_pq_reasoning
                            })
                        if self.enable_sc_metric:
                            sample_detail.update({
                                "primary_sc_score": pair.primary_sc_score,
                                "primary_sc_reasoning": pair.primary_sc_reasoning
                            })
                        if self.enable_instruction_following_metric:
                            sample_detail.update({
                                "primary_if_score": pair.primary_if_score,
                                "primary_if_reasoning": pair.primary_if_reasoning
                            })
                    else:
                        # 正常模式：包含 refined 和 primary
                        sample_detail.update({
                            "refined_score": pair.refined_score,
                            "refined_score_reasoning": pair.refined_score_reasoning
                        })
                        if self.enable_primary_scoring:
                            sample_detail.update({
                                "primary_score": pair.primary_score,
                                "primary_score_reasoning": pair.primary_score_reasoning,
                                "improvement_rate": pair.improvement_rate
                            })
                        if self.enable_pq_metric:
                            pq_detail = {
                                "refined_pq_score": pair.refined_pq_score,
                                "refined_pq_reasoning": pair.refined_pq_reasoning
                            }
                            if self.enable_primary_scoring:
                                pq_detail.update({
                                    "primary_pq_score": pair.primary_pq_score,
                                    "primary_pq_reasoning": pair.primary_pq_reasoning
                                })
                            sample_detail.update(pq_detail)
                        if self.enable_sc_metric:
                            sc_detail = {
                                "refined_sc_score": pair.refined_sc_score,
                                "refined_sc_reasoning": pair.refined_sc_reasoning
                            }
                            if self.enable_primary_scoring:
                                sc_detail.update({
                                    "primary_sc_score": pair.primary_sc_score,
                                    "primary_sc_reasoning": pair.primary_sc_reasoning
                                })
                            sample_detail.update(sc_detail)
                        if self.enable_instruction_following_metric:
                            if_detail = {
                                "refined_if_score": pair.refined_if_score,
                                "refined_if_reasoning": pair.refined_if_reasoning
                            }
                            if self.enable_primary_scoring:
                                if_detail.update({
                                    "primary_if_score": pair.primary_if_score,
                                    "primary_if_reasoning": pair.primary_if_reasoning
                                })
                            sample_detail.update(if_detail)
                
                mllm_analysis_details[category_name].append(sample_detail)
        
        # 构建overall statistics（兼容standard pipeline）
        if not self.skip_stage4:
            if self.skip_refinement:
                overall_statistics = {
                    "global_primary_yes_count": global_stats.get('global_primary_yes_count', 0),
                    "global_primary_yes_rate": global_stats.get('global_primary_yes_rate', 0.0),
                    "average": global_stats.get('global_primary_avg', 0.0),
                    "std": 0.0,
                    "total_samples": global_stats['total_pairs'],
                    "primary_average": global_stats.get('global_primary_avg', 0.0)
                }
            else:
                overall_statistics = {
                    "global_refined_yes_count": global_stats.get('global_refined_yes_count', 0),
                    "global_refined_yes_rate": global_stats.get('global_refined_yes_rate', 0.0),
                    "average": global_stats.get('global_refined_avg', 0.0),
                    "std": 0.0,
                    "total_samples": global_stats['total_pairs'],
                    "refined_average": global_stats.get('global_refined_avg', 0.0)
                }
            
            # Primary统计（非 skip_refinement 时，enable_primary_scoring 时包含）
            if self.enable_primary_scoring and not self.skip_refinement:
                overall_statistics.update({
                    "global_primary_yes_count": global_stats.get('global_primary_yes_count', 0),
                    "global_primary_yes_rate": global_stats.get('global_primary_yes_rate', 0.0),
                    "primary_average": global_stats.get('global_primary_avg', 0.0),
                    # 改进统计
                    "global_improved_count": global_stats.get('global_improved_count', 0),
                    "global_improvement_rate": global_stats.get('global_improvement_rate', 0.0),
                    "global_maintained_count": global_stats.get('global_maintained_count', 0),
                    "global_maintained_rate": global_stats.get('global_maintained_rate', 0.0),
                    "global_regression_count": global_stats.get('global_regression_count', 0),
                    "global_regression_rate": global_stats.get('global_regression_rate', 0.0),
                    "global_unchanged_count": global_stats.get('global_unchanged_count', 0),
                    "global_unchanged_rate": global_stats.get('global_unchanged_rate', 0.0)
                })
            
            # PQ指标全局统计（只在启用时包含）
            if self.enable_pq_metric:
                if self.skip_refinement:
                    pq_overall = {"global_primary_pq_avg": global_stats.get('global_primary_pq_avg', 0.0)}
                else:
                    pq_overall = {
                        "global_refined_pq_avg": global_stats.get('global_refined_pq_avg', 0.0)
                    }
                    if self.enable_primary_scoring:
                        pq_overall.update({
                            "global_primary_pq_avg": global_stats.get('global_primary_pq_avg', 0.0),
                            "global_pq_improvement": global_stats.get('global_pq_improvement', 0.0)
                        })
                overall_statistics.update(pq_overall)
            
            # SC指标全局统计（只在启用时包含）
            if self.enable_sc_metric:
                if self.skip_refinement:
                    sc_overall = {"global_primary_sc_avg": global_stats.get('global_primary_sc_avg', 0.0)}
                else:
                    sc_overall = {
                        "global_refined_sc_avg": global_stats.get('global_refined_sc_avg', 0.0)
                    }
                    if self.enable_primary_scoring:
                        sc_overall.update({
                            "global_primary_sc_avg": global_stats.get('global_primary_sc_avg', 0.0),
                            "global_sc_improvement": global_stats.get('global_sc_improvement', 0.0)
                        })
                overall_statistics.update(sc_overall)
            
            # IF指标全局统计（只在启用时包含）
            if self.enable_instruction_following_metric:
                if self.skip_refinement:
                    if_overall = {"global_primary_if_avg": global_stats.get('global_primary_if_avg', 0.0)}
                else:
                    if_overall = {
                        "global_refined_if_avg": global_stats.get('global_refined_if_avg', 0.0)
                    }
                    if self.enable_primary_scoring:
                        if_overall.update({
                            "global_primary_if_avg": global_stats.get('global_primary_if_avg', 0.0),
                            "global_if_improvement": global_stats.get('global_if_improvement', 0.0)
                        })
                overall_statistics.update(if_overall)
        else:
            # 如果跳过 Stage4，不包含评价相关的统计信息
            overall_statistics = {
                "total_samples": global_stats['total_pairs'],
                "note": "Scoring statistics not available (Stage 4 was skipped)"
            }
        
        # 构建metadata
        metadata = {
            "pipeline_mode": "iterative_refinement",
            "benchmark_config": self.config.get("benchmark", {}),
            "diffusion_model": self.config.get("diffusion_model", {}),
            "mllm": self.config.get("mllm", {}),
            "refinement_diffusion": self.config.get("diffusion_model", {}).get("refinement", {}),
            "reward_model": self.config.get("reward_model", {}),
            "total_pairs": global_stats['total_pairs'],
            "categories": benchmark_data.category_names,
            # 评分指标配置
            "evaluation_config": {
                "enable_pq_metric": self.enable_pq_metric,
                "enable_sc_metric": self.enable_sc_metric,
                "enable_instruction_following_metric": self.enable_instruction_following_metric,
                "enable_primary_scoring": self.enable_primary_scoring,
                "skip_stage4": self.skip_stage4,
                "skip_refinement": self.skip_refinement  # 是否跳过 refinement 流程
            }
        }
        
        # 构建summary（兼容standard pipeline的main.py）
        if not self.skip_stage4:
            if self.skip_refinement:
                summary = {
                    "num_categories": len(category_statistics),
                    "total_samples": global_stats['total_pairs'],
                    "overall_primary_yes_rate": global_stats.get('global_primary_yes_rate', 0.0),
                    "overall_mean": global_stats.get('global_primary_avg', 0.0),
                    "category_means": {
                        cat_name: cat_stats.get("primary_avg", 0.0)
                        for cat_name, cat_stats in category_statistics.items()
                        if "primary_avg" in cat_stats
                    },
                    "category_yes_rates": {
                        cat_name: cat_stats.get("primary_yes_rate", 0.0)
                        for cat_name, cat_stats in category_statistics.items()
                        if "primary_yes_rate" in cat_stats
                    },
                    "primary_overall_mean": global_stats.get('global_primary_avg', 0.0)
                }
            else:
                summary = {
                    "num_categories": len(category_statistics),
                    "total_samples": global_stats['total_pairs'],
                    "overall_refined_yes_rate": global_stats.get('global_refined_yes_rate', 0.0),
                    "overall_mean": global_stats.get('global_refined_avg', 0.0),
                    "category_means": {
                        cat_name: cat_stats.get("refined_avg", 0.0)
                        for cat_name, cat_stats in category_statistics.items()
                        if "refined_avg" in cat_stats
                    },
                    "category_yes_rates": {
                        cat_name: cat_stats.get("refined_yes_rate", 0.0)
                        for cat_name, cat_stats in category_statistics.items()
                        if "refined_yes_rate" in cat_stats
                    },
                    "refined_overall_mean": global_stats.get('global_refined_avg', 0.0)
                }
            
            # Primary统计（非 skip_refinement 时，enable_primary_scoring 时包含改进统计）
            if self.enable_primary_scoring and not self.skip_refinement:
                summary.update({
                    "overall_primary_yes_rate": global_stats.get('global_primary_yes_rate', 0.0),
                    "primary_overall_mean": global_stats.get('global_primary_avg', 0.0),
                    "category_primary_yes_rates": {
                        cat_name: cat_stats.get("primary_yes_rate", 0.0)
                        for cat_name, cat_stats in category_statistics.items()
                        if "primary_yes_rate" in cat_stats
                    },
                    "overall_improvement_rate": global_stats.get('global_improvement_rate', 0.0),
                    "overall_maintained_rate": global_stats.get('global_maintained_rate', 0.0),
                    "overall_regression_rate": global_stats.get('global_regression_rate', 0.0),
                    "overall_unchanged_rate": global_stats.get('global_unchanged_rate', 0.0),
                    "total_improved": global_stats.get('global_improved_count', 0),
                    "improvement_percentage": global_stats.get('global_improvement_rate', 0.0)
                })
            
            # PQ指标汇总（只在启用时包含）
            if self.enable_pq_metric:
                if self.skip_refinement:
                    pq_summary = {
                        "overall_primary_pq_avg": global_stats.get('global_primary_pq_avg', 0.0),
                        "category_pq_primary_avgs": {
                            cat_name: cat_stats.get("primary_pq_avg", 0.0)
                            for cat_name, cat_stats in category_statistics.items()
                            if "primary_pq_avg" in cat_stats
                        }
                    }
                else:
                    pq_summary = {
                        "overall_refined_pq_avg": global_stats.get('global_refined_pq_avg', 0.0),
                        "category_pq_refined_avgs": {
                            cat_name: cat_stats.get("refined_pq_avg", 0.0)
                            for cat_name, cat_stats in category_statistics.items()
                            if "refined_pq_avg" in cat_stats
                        }
                    }
                    if self.enable_primary_scoring:
                        pq_summary.update({
                            "overall_primary_pq_avg": global_stats.get('global_primary_pq_avg', 0.0),
                            "category_pq_primary_avgs": {
                                cat_name: cat_stats.get("primary_pq_avg", 0.0)
                                for cat_name, cat_stats in category_statistics.items()
                                if "primary_pq_avg" in cat_stats
                            },
                            "overall_pq_improvement": global_stats.get('global_pq_improvement', 0.0)
                        })
                summary.update(pq_summary)
            
            # SC指标汇总（只在启用时包含）
            if self.enable_sc_metric:
                if self.skip_refinement:
                    sc_summary = {
                        "overall_primary_sc_avg": global_stats.get('global_primary_sc_avg', 0.0),
                        "category_sc_primary_avgs": {
                            cat_name: cat_stats.get("primary_sc_avg", 0.0)
                            for cat_name, cat_stats in category_statistics.items()
                            if "primary_sc_avg" in cat_stats
                        }
                    }
                else:
                    sc_summary = {
                        "overall_refined_sc_avg": global_stats.get('global_refined_sc_avg', 0.0),
                        "category_sc_refined_avgs": {
                            cat_name: cat_stats.get("refined_sc_avg", 0.0)
                            for cat_name, cat_stats in category_statistics.items()
                            if "refined_sc_avg" in cat_stats
                        }
                    }
                    if self.enable_primary_scoring:
                        sc_summary.update({
                            "overall_primary_sc_avg": global_stats.get('global_primary_sc_avg', 0.0),
                            "category_sc_primary_avgs": {
                                cat_name: cat_stats.get("primary_sc_avg", 0.0)
                                for cat_name, cat_stats in category_statistics.items()
                                if "primary_sc_avg" in cat_stats
                            },
                            "overall_sc_improvement": global_stats.get('global_sc_improvement', 0.0)
                        })
                summary.update(sc_summary)
            
            # IF指标汇总（只在启用时包含）
            if self.enable_instruction_following_metric:
                if self.skip_refinement:
                    if_summary = {
                        "overall_primary_if_avg": global_stats.get('global_primary_if_avg', 0.0),
                        "category_if_primary_avgs": {
                            cat_name: cat_stats.get("primary_if_avg", 0.0)
                            for cat_name, cat_stats in category_statistics.items()
                            if "primary_if_avg" in cat_stats
                        }
                    }
                else:
                    if_summary = {
                        "overall_refined_if_avg": global_stats.get('global_refined_if_avg', 0.0),
                        "category_if_refined_avgs": {
                            cat_name: cat_stats.get("refined_if_avg", 0.0)
                            for cat_name, cat_stats in category_statistics.items()
                            if "refined_if_avg" in cat_stats
                        }
                    }
                    if self.enable_primary_scoring:
                        if_summary.update({
                            "overall_primary_if_avg": global_stats.get('global_primary_if_avg', 0.0),
                            "category_if_primary_avgs": {
                                cat_name: cat_stats.get("primary_if_avg", 0.0)
                                for cat_name, cat_stats in category_statistics.items()
                                if "primary_if_avg" in cat_stats
                            },
                            "overall_if_improvement": global_stats.get('global_if_improvement', 0.0)
                        })
                summary.update(if_summary)
            
            # 找出最好和最差的类别（基于refined yes_rate）
            if summary.get("category_means"):
                best_category_name = max(summary["category_means"], key=summary["category_means"].get)
                worst_category_name = min(summary["category_means"], key=summary["category_means"].get)
                
                summary["best_category"] = {
                    "name": best_category_name,
                    "score": summary["category_means"][best_category_name]  # yes_rate
                }
                summary["worst_category"] = {
                    "name": worst_category_name,
                    "score": summary["category_means"][worst_category_name]  # yes_rate
                }
            
            # Primary类别排名（skip_refinement 时 category_means 已是 primary；非 skip_refinement 时 enable_primary_scoring 包含）
            if self.skip_refinement:
                pass  # best_category 已基于 primary
            elif self.enable_primary_scoring and summary.get("category_primary_yes_rates") and len(summary["category_primary_yes_rates"]) > 0:
                best_primary_category_name = max(summary["category_primary_yes_rates"], key=summary["category_primary_yes_rates"].get)
                worst_primary_category_name = min(summary["category_primary_yes_rates"], key=summary["category_primary_yes_rates"].get)
                
                summary["best_primary_category"] = {
                    "name": best_primary_category_name,
                    "score": summary["category_primary_yes_rates"][best_primary_category_name]
                }
                summary["worst_primary_category"] = {
                    "name": worst_primary_category_name,
                    "score": summary["category_primary_yes_rates"][worst_primary_category_name]
                }
            
            # 找出 PQ 指标最好和最差的类别
            pq_avgs = summary.get("category_pq_primary_avgs") if self.skip_refinement else summary.get("category_pq_refined_avgs")
            if pq_avgs and len(pq_avgs) > 0:
                best_pq_category_name = max(pq_avgs, key=pq_avgs.get)
                worst_pq_category_name = min(pq_avgs, key=pq_avgs.get)
                summary["best_pq_category"] = {
                    "name": best_pq_category_name,
                    "score": pq_avgs[best_pq_category_name]
                }
                summary["worst_pq_category"] = {
                    "name": worst_pq_category_name,
                    "score": pq_avgs[worst_pq_category_name]
                }
                
                if self.skip_refinement:
                    summary["best_pq_primary_category"] = summary["best_pq_category"]
                    summary["worst_pq_primary_category"] = summary["worst_pq_category"]
                elif self.enable_primary_scoring and summary.get("category_pq_primary_avgs") and len(summary["category_pq_primary_avgs"]) > 0:
                    best_pq_primary_category_name = max(summary["category_pq_primary_avgs"], key=summary["category_pq_primary_avgs"].get)
                    worst_pq_primary_category_name = min(summary["category_pq_primary_avgs"], key=summary["category_pq_primary_avgs"].get)
                    
                    summary["best_pq_primary_category"] = {
                        "name": best_pq_primary_category_name,
                        "score": summary["category_pq_primary_avgs"][best_pq_primary_category_name]
                    }
                    summary["worst_pq_primary_category"] = {
                        "name": worst_pq_primary_category_name,
                        "score": summary["category_pq_primary_avgs"][worst_pq_primary_category_name]
                    }
            
            # 找出 SC 指标最好和最差的类别
            sc_avgs = summary.get("category_sc_primary_avgs") if self.skip_refinement else summary.get("category_sc_refined_avgs")
            if sc_avgs and len(sc_avgs) > 0:
                best_sc_category_name = max(sc_avgs, key=sc_avgs.get)
                worst_sc_category_name = min(sc_avgs, key=sc_avgs.get)
                summary["best_sc_category"] = {
                    "name": best_sc_category_name,
                    "score": sc_avgs[best_sc_category_name]
                }
                summary["worst_sc_category"] = {
                    "name": worst_sc_category_name,
                    "score": sc_avgs[worst_sc_category_name]
                }
                
                if self.skip_refinement:
                    summary["best_sc_primary_category"] = summary["best_sc_category"]
                    summary["worst_sc_primary_category"] = summary["worst_sc_category"]
                elif self.enable_primary_scoring and summary.get("category_sc_primary_avgs") and len(summary["category_sc_primary_avgs"]) > 0:
                    best_sc_primary_category_name = max(summary["category_sc_primary_avgs"], key=summary["category_sc_primary_avgs"].get)
                    worst_sc_primary_category_name = min(summary["category_sc_primary_avgs"], key=summary["category_sc_primary_avgs"].get)
                    
                    summary["best_sc_primary_category"] = {
                        "name": best_sc_primary_category_name,
                        "score": summary["category_sc_primary_avgs"][best_sc_primary_category_name]
                    }
                    summary["worst_sc_primary_category"] = {
                        "name": worst_sc_primary_category_name,
                        "score": summary["category_sc_primary_avgs"][worst_sc_primary_category_name]
                    }
            
            # 找出 IF 指标最好和最差的类别
            if_avgs = summary.get("category_if_primary_avgs") if self.skip_refinement else summary.get("category_if_refined_avgs")
            if if_avgs and len(if_avgs) > 0:
                best_if_category_name = max(if_avgs, key=if_avgs.get)
                worst_if_category_name = min(if_avgs, key=if_avgs.get)
                summary["best_if_category"] = {
                    "name": best_if_category_name,
                    "score": if_avgs[best_if_category_name]
                }
                summary["worst_if_category"] = {
                    "name": worst_if_category_name,
                    "score": if_avgs[worst_if_category_name]
                }
                
                if self.skip_refinement:
                    summary["best_if_primary_category"] = summary["best_if_category"]
                    summary["worst_if_primary_category"] = summary["worst_if_category"]
                elif self.enable_primary_scoring and summary.get("category_if_primary_avgs") and len(summary["category_if_primary_avgs"]) > 0:
                    best_if_primary_category_name = max(summary["category_if_primary_avgs"], key=summary["category_if_primary_avgs"].get)
                    worst_if_primary_category_name = min(summary["category_if_primary_avgs"], key=summary["category_if_primary_avgs"].get)
                    
                    summary["best_if_primary_category"] = {
                        "name": best_if_primary_category_name,
                        "score": summary["category_if_primary_avgs"][best_if_primary_category_name]
                    }
                    summary["worst_if_primary_category"] = {
                        "name": worst_if_primary_category_name,
                        "score": summary["category_if_primary_avgs"][worst_if_primary_category_name]
                    }
        else:
            # 如果跳过 Stage4，只包含基础信息
            summary = {
                "num_categories": len(category_statistics),
                "total_samples": global_stats['total_pairs'],
                "note": "Scoring statistics not available (Stage 4 was skipped)"
            }
        
        # 组装完整报告
        # 生成报告文件名（如果配置了 use_original_as_primary、primary_images_dir 或 skip_stage4，添加标识以区分）
        filename_parts = []
        
        if self.use_original_as_primary:
            filename_parts.append("use_original_as_primary")
        elif self.primary_images_dir:
            primary_dir_name = os.path.basename(os.path.normpath(self.primary_images_dir))
            filename_parts.append(f"refined_only_{primary_dir_name}")
        
        if self.skip_stage4:
            filename_parts.append("skip_stage4")
        if self.skip_refinement:
            filename_parts.append("skip_refinement")
        
        if filename_parts:
            report_filename = f"{self.primary_model_type}_{'_'.join(filename_parts)}_{self.start_timestamp}.json"
        else:
            report_filename = f"{self.primary_model_type}_{self.start_timestamp}.json"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # 计算scoring_health统计（如果未跳过 Stage4）
        if not self.skip_stage4:
            scoring_health_summary = self._compute_scoring_health_summary()
        else:
            scoring_health_summary = {
                "note": "Scoring health statistics not available (Stage 4 was skipped)"
            }
        
        report = {
            "metadata": metadata,
            "category_statistics": category_statistics,
            "overall_statistics": overall_statistics,
            "summary": summary,  # 兼容standard pipeline
            "mllm_analysis": mllm_analysis_details,  # Iterative特有：MLLM分析详情
            "scoring_health": scoring_health_summary,  # 方案3：评分健康度追踪（如果未跳过 Stage4）
            "timestamp": self._get_timestamp(),
            "report_file": {
                "filename": report_filename,
                "path": report_path,
                "model_type": self.primary_model_type,
                "start_time": self.start_timestamp
            }
        }
        
        # 保存统计信息到文件（使用动态文件名：模型类型_时间戳.json）
        import json
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Report saved to: " + report_path)
        self.logger.info(f"Report filename: {report_filename}")
        self.logger.info(f"  Model Type: {self.primary_model_type}")
        self.logger.info(f"  Start Time: {self.start_timestamp}")
        
        # 输出MLLM分析统计
        self.logger.info(f"\n{'='*80}")
        self.logger.info("MLLM Analysis Summary:")
        self.logger.info(f"{'='*80}")
        
        total_samples = 0
        samples_with_cot = 0
        samples_with_different_instruction = 0
        
        for category_name, samples in mllm_analysis_details.items():
            for sample in samples:
                total_samples += 1
                if sample["cot_reasoning"] and len(sample["cot_reasoning"].strip()) > 0:
                    samples_with_cot += 1
                # 检查re-edit指令是否与原始指令不同
                if sample["re_edit_instruction"] != sample["original_instruction"]:
                    samples_with_different_instruction += 1
        
        self.logger.info(f"Total Samples Analyzed: {total_samples}")
        self.logger.info(f"Samples with CoT Reasoning: {samples_with_cot}/{total_samples} ({samples_with_cot/total_samples*100:.1f}%)")
        self.logger.info(f"Samples with Modified Instructions: {samples_with_different_instruction}/{total_samples} ({samples_with_different_instruction/total_samples*100:.1f}%)")
        self.logger.info(f"{'='*80}")
        
        # 输出Scoring Health Summary（如果未跳过 Stage4）
        if not self.skip_stage4:
            self.logger.info(f"\n{'='*80}")
            self.logger.info("Scoring Health Summary:")
            self.logger.info(f"{'='*80}")
            
            health = scoring_health_summary
            if 'overall_status' in health:
                self.logger.info(f"Overall Status: {health['overall_status'].upper()}")
                
                # Primary Scoring（只在enable_primary_scoring时输出）
                if self.enable_primary_scoring and 'primary_scoring' in health:
                    primary = health['primary_scoring']
                    self.logger.info(f"\nPrimary Scoring:")
                    self.logger.info(f"  Status: {primary['status'].upper()}")
                    self.logger.info(f"  Total Samples: {primary['total_samples']}")
                    self.logger.info(f"  Failed Samples: {primary['failed_samples']} ({primary['failure_rate']:.1f}%)")
                    if primary.get('failed_gpus'):
                        self.logger.info(f"  Failed GPUs: {primary['failed_gpus']}")
                
                # Refined Scoring
                if 'refined_scoring' in health:
                    refined = health['refined_scoring']
                    self.logger.info(f"\nRefined Scoring:")
                    self.logger.info(f"  Status: {refined['status'].upper()}")
                    self.logger.info(f"  Total Samples: {refined['total_samples']}")
                    self.logger.info(f"  Failed Samples: {refined['failed_samples']} ({refined['failure_rate']:.1f}%)")
                    if refined.get('failed_gpus'):
                        self.logger.info(f"  Failed GPUs: {refined['failed_gpus']}")
                
                # Warning Message
                if health.get('warning_message'):
                    self.logger.warning(f"\n⚠️  {health['warning_message']}")
            else:
                self.logger.info(health.get('note', 'Scoring health statistics not available'))
            
            self.logger.info(f"{'='*80}")
        
        return report
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _compute_scoring_health_summary(self) -> Dict[str, Any]:
        """
        计算评分健康度汇总统计（包含 primary 和 refined 评分）
        
        Returns:
            评分健康度汇总字典
        """
        # 判断健康状态辅助函数
        def get_health_status(failure_rate):
            if failure_rate == 0:
                return "healthy"
            elif failure_rate < 10:
                return "warning"
            elif failure_rate < 50:
                return "degraded"
            else:
                return "critical"
        
        # ===== Primary评分健康度（只在enable_primary_scoring时计算）=====
        if self.enable_primary_scoring:
            # 合并所有失败的GPU（去重）
            all_failed_gpus_primary = set()
            for category, failures in self.scoring_health['primary_failures'].items():
                all_failed_gpus_primary.update(failures['failed_gpus'])
            
            # 计算失败百分比
            primary_failure_rate = 0.0
            if self.scoring_health['total_primary_samples'] > 0:
                primary_failure_rate = (self.scoring_health['total_primary_failures'] / 
                                       self.scoring_health['total_primary_samples'] * 100)
            
            primary_status = get_health_status(primary_failure_rate)
        
        # ===== Refined评分健康度（总是计算）=====
        # 合并所有失败的GPU（去重）
        all_failed_gpus_refined = set()
        for category, failures in self.scoring_health['refined_failures'].items():
            all_failed_gpus_refined.update(failures['failed_gpus'])
        
        # 计算失败百分比
        refined_failure_rate = 0.0
        if self.scoring_health['total_refined_samples'] > 0:
            refined_failure_rate = (self.scoring_health['total_refined_failures'] / 
                                   self.scoring_health['total_refined_samples'] * 100)
        
        refined_status = get_health_status(refined_failure_rate)
        
        # ===== 构建详细的失败信息（按类别）=====
        category_details = {}
        for category in self.scoring_health['refined_failures'].keys():
            refined_fail = self.scoring_health['refined_failures'].get(category, {})
            category_detail = {
                "refined_scoring": {
                    "total_samples": refined_fail.get('total_samples', 0),
                    "failed_samples": refined_fail.get('failed_sample_count', 0),
                    "failure_rate": (refined_fail.get('failed_sample_count', 0) / 
                                    refined_fail.get('total_samples', 1) * 100) if refined_fail.get('total_samples', 0) > 0 else 0.0,
                    "failed_gpus": refined_fail.get('failed_gpus', [])
                }
            }
            
            # 添加primary失败信息（只在enable_primary_scoring时）
            if self.enable_primary_scoring:
                primary_fail = self.scoring_health['primary_failures'].get(category, {})
                category_detail["primary_scoring"] = {
                    "total_samples": primary_fail.get('total_samples', 0),
                    "failed_samples": primary_fail.get('failed_sample_count', 0),
                    "failure_rate": (primary_fail.get('failed_sample_count', 0) / 
                                    primary_fail.get('total_samples', 1) * 100) if primary_fail.get('total_samples', 0) > 0 else 0.0,
                    "failed_gpus": primary_fail.get('failed_gpus', [])
                }
            
            category_details[category] = category_detail
        
        # ===== 构建汇总信息 =====
        # 综合状态：取primary和refined中较差的状态
        if self.enable_primary_scoring:
            status_priority = {"healthy": 0, "warning": 1, "degraded": 2, "critical": 3}
            overall_status = max([primary_status, refined_status], key=lambda s: status_priority.get(s, 0))
        else:
            overall_status = refined_status
        
        summary = {
            "overall_status": overall_status,
            "refined_scoring": {
                "status": refined_status,
                "total_samples": self.scoring_health['total_refined_samples'],
                "failed_samples": self.scoring_health['total_refined_failures'],
                "failure_rate": round(refined_failure_rate, 2),
                "failed_gpus": sorted(list(all_failed_gpus_refined))
            },
            "category_details": category_details,
            "warning_message": self._generate_health_warning_message(
                primary_failure_rate if self.enable_primary_scoring else None,
                refined_failure_rate
            )
        }
        
        # 添加primary健康度（只在enable_primary_scoring时）
        if self.enable_primary_scoring:
            summary["primary_scoring"] = {
                "status": primary_status,
                "total_samples": self.scoring_health['total_primary_samples'],
                "failed_samples": self.scoring_health['total_primary_failures'],
                "failure_rate": round(primary_failure_rate, 2),
                "failed_gpus": sorted(list(all_failed_gpus_primary))
            }
        
        return summary
    
    def _generate_health_warning_message(self, primary_failure_rate: Optional[float], refined_failure_rate: float) -> Optional[str]:
        """
        生成健康警告信息
        
        Args:
            primary_failure_rate: Primary评分失败率（可选，只在enable_primary_scoring时提供）
            refined_failure_rate: Refined评分失败率
        
        Returns:
            警告信息，如果无问题则返回None
        """
        messages = []
        
        # Primary失败率检查（只在enable_primary_scoring时）
        if primary_failure_rate is not None:
            if primary_failure_rate >= 50:
                messages.append(f"CRITICAL: Primary scoring failed for {primary_failure_rate:.1f}% of samples (possible CUDA OOM)")
            elif primary_failure_rate >= 10:
                messages.append(f"WARNING: Primary scoring failed for {primary_failure_rate:.1f}% of samples")
            elif primary_failure_rate > 0:
                messages.append(f"Note: Primary scoring failed for {primary_failure_rate:.1f}% of samples")
        
        # Refined失败率检查
        if refined_failure_rate >= 50:
            messages.append(f"CRITICAL: Refined scoring failed for {refined_failure_rate:.1f}% of samples (possible CUDA OOM)")
        elif refined_failure_rate >= 10:
            messages.append(f"WARNING: Refined scoring failed for {refined_failure_rate:.1f}% of samples")
        elif refined_failure_rate > 0:
            messages.append(f"Note: Refined scoring failed for {refined_failure_rate:.1f}% of samples")
        
        if messages:
            messages.append("Failed samples are assigned default result 'no'")
            return " | ".join(messages)
        
        return None

