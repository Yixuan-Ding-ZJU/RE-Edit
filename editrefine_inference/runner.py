"""
EditRefine 单条推理：Primary 编辑 → MLLM 分析（通用 prompt）→ 单步 Refinement → 保存 4 项输出
复用 config 中的 diffusion_model（primary/refinement）与 mllm；MLLM 使用通用维度（physical 作为通用 prompt）
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

# 假定由 run_editrefine_inference.py 在项目根将 project root 加入 sys.path，此处从 src 导入

logger = logging.getLogger(__name__)

# 通用 category：MLLM 五类 prompt 内容一致，统一用 physical
DEFAULT_CATEGORY = "physical"


def _load_primary_diffusion(config: dict):
    """与 iterative_pipeline_v7 一致的 primary diffusion 加载逻辑"""
    diffusion_config = config["diffusion_model"]["primary"]
    model_type = diffusion_config["type"]
    model_params = diffusion_config.get("params", {})

    if model_type == "multi_gpu_qwen_edit":
        from src.models.diffusion.implementations.qwen_image_edit import QwenImageEditModel
        return QwenImageEditModel(model_params)
    elif model_type == "qwen_image_edit_2511":
        from src.models.diffusion.implementations.qwen_image_edit_2511 import QwenImageEdit2511Model
        return QwenImageEdit2511Model(model_params)
    elif model_type == "flux_kontext":
        from src.models.diffusion.implementations.flux_kontext import FluxKontextModel
        return FluxKontextModel(model_params)
    elif model_type == "step1x_edit_v1p2_preview":
        from src.models.diffusion.implementations.step1x_edit_v1p2_preview import Step1XEditModel
        return Step1XEditModel(model_params)
    elif model_type == "step1x_edit_v1p1":
        from src.models.diffusion.implementations.step1x_edit_v1p1 import Step1XEditPreviewModel
        return Step1XEditPreviewModel(model_params)
    elif model_type == "dreamomni2":
        from src.models.diffusion.implementations.dreamomni2 import DreamOmni2Model
        return DreamOmni2Model(model_params)
    elif model_type == "omnigen2":
        from src.models.diffusion.implementations.omnigen2 import OmniGen2Model
        return OmniGen2Model(model_params)
    elif model_type == "hidream_e1":
        from src.models.diffusion.implementations.hidream_e1 import HiDreamE1Model
        return HiDreamE1Model(model_params)
    elif model_type == "ovis_u1":
        from src.models.diffusion.implementations.ovis_u1 import OvisU1Model
        return OvisU1Model(model_params)
    elif model_type == "janus":
        from src.models.diffusion.implementations.janus import JanusModel
        return JanusModel(model_params)
    elif model_type == "flux2_dev":
        from src.models.diffusion.implementations.flux2_dev import Flux2DevModel
        return Flux2DevModel(model_params)
    else:
        raise ValueError(f"Unknown primary diffusion model type: {model_type}")


def _load_refinement_diffusion(config: dict):
    """与 iterative_pipeline_v7 一致的 refinement diffusion 加载逻辑"""
    refinement_config = config["diffusion_model"].get("refinement")
    if not refinement_config:
        return None
    model_type = refinement_config["type"]
    model_params = refinement_config.get("params", {})

    if model_type == "multi_gpu_qwen_edit":
        from src.models.diffusion.implementations.qwen_image_edit import QwenImageEditModel
        return QwenImageEditModel(model_params)
    elif model_type == "qwen_image_edit_2511":
        from src.models.diffusion.implementations.qwen_image_edit_2511 import QwenImageEdit2511Model
        return QwenImageEdit2511Model(model_params)
    elif model_type == "flux_kontext":
        from src.models.diffusion.implementations.flux_kontext import FluxKontextModel
        return FluxKontextModel(model_params)
    elif model_type == "step1x_edit_v1p2_preview":
        from src.models.diffusion.implementations.step1x_edit_v1p2_preview import Step1XEditModel
        return Step1XEditModel(model_params)
    elif model_type == "step1x_edit_v1p1":
        from src.models.diffusion.implementations.step1x_edit_v1p1 import Step1XEditPreviewModel
        return Step1XEditPreviewModel(model_params)
    elif model_type == "dreamomni2":
        from src.models.diffusion.implementations.dreamomni2 import DreamOmni2Model
        return DreamOmni2Model(model_params)
    elif model_type == "omnigen2":
        from src.models.diffusion.implementations.omnigen2 import OmniGen2Model
        return OmniGen2Model(model_params)
    elif model_type == "hidream_e1":
        from src.models.diffusion.implementations.hidream_e1 import HiDreamE1Model
        return HiDreamE1Model(model_params)
    elif model_type == "ovis_u1":
        from src.models.diffusion.implementations.ovis_u1 import OvisU1Model
        return OvisU1Model(model_params)
    elif model_type == "janus":
        from src.models.diffusion.implementations.janus import JanusModel
        return JanusModel(model_params)
    elif model_type == "flux2_dev":
        from src.models.diffusion.implementations.flux2_dev import Flux2DevModel
        return Flux2DevModel(model_params)
    else:
        raise ValueError(f"Unknown refinement diffusion model type: {model_type}")


def _load_mllm(config: dict):
    """与 iterative_pipeline_v7 一致的 MLLM 加载逻辑"""
    mllm_config = config["mllm"]
    model_type = mllm_config["type"]
    model_params = mllm_config.get("params", {})
    if model_type == "qwen25_vl":
        from src.models.mllm.implementations.qwen25_vl_mllm import Qwen25VLMllm
        return Qwen25VLMllm(model_params)
    else:
        raise ValueError(f"Unknown MLLM type: {model_type}")


def run_single(
    config: dict,
    image: Image.Image,
    edit_instruction: str,
    original_description: str = "",
    output_dir: Optional[str] = None,
    output_name_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    单条 EditRefine 推理：原图 + 编辑指令 → Primary → MLLM → 单步 Refinement，并保存 4 项输出。

    Args:
        config: 合并后的配置（含 diffusion_model、mllm、editrefine）
        image: 原图 PIL Image
        edit_instruction: 编辑指令
        original_description: 原图描述（可选，MLLM 可用空串）
        output_dir: 输出目录，默认用 config["editrefine"]["output_dir"]
        output_name_prefix: 输出文件名前缀，默认用 "editrefine"

    Returns:
        {
            "primary_image": PIL.Image,
            "refined_image": PIL.Image,
            "cot": str,
            "re_edit_instruction": str,
            "paths": {"primary": str, "refined": str, "cot": str, "re_edit": str}
        }
    """
    editrefine = config.get("editrefine", {})
    out_dir = output_dir or editrefine.get("output_dir", "./editrefine_inference_output")
    prefix = output_name_prefix if output_name_prefix is not None else editrefine.get("output_name_prefix") or "editrefine"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Primary 编辑（模型 __init__ 中已调用 _initialize）
    logger.info("Loading primary diffusion model...")
    primary_model = _load_primary_diffusion(config)
    if hasattr(primary_model, "load_to_gpu"):
        primary_model.load_to_gpu()
    primary_images = primary_model.batch_edit([image], [edit_instruction])
    primary_image = primary_images[0]
    if hasattr(primary_model, "unload_from_gpu"):
        primary_model.unload_from_gpu()

    # 2) MLLM 分析（通用 prompt：使用 physical）
    logger.info("Loading MLLM...")
    mllm = _load_mllm(config)
    mllm._load_model()
    mllm_outputs = mllm.batch_analyze(
        edited_images=[primary_image],
        original_images=[image],
        original_descriptions=[original_description],
        edit_instructions=[edit_instruction],
        categories=[DEFAULT_CATEGORY],
        batch_size=1,
    )
    out = mllm_outputs[0]
    cot = out.get("cot", "")
    re_edit_instruction = out.get("re_edit_instruction", "").strip()
    if not re_edit_instruction and out.get("re_edit_instructions"):
        re_edit_instruction = " ".join(out["re_edit_instructions"]).strip()
    if not re_edit_instruction:
        re_edit_instruction = "Improve the image quality"

    # 3) 单步 Refinement
    refinement_config = config["diffusion_model"].get("refinement")
    if refinement_config:
        logger.info("Loading refinement diffusion model...")
        refinement_model = _load_refinement_diffusion(config)
        if hasattr(refinement_model, "load_to_gpu"):
            refinement_model.load_to_gpu()
        refined_images = refinement_model.batch_edit([primary_image], [re_edit_instruction])
        refined_image = refined_images[0]
        if hasattr(refinement_model, "unload_from_gpu"):
            refinement_model.unload_from_gpu()
    else:
        refined_image = primary_image
        logger.info("No refinement model configured, using primary image as refined output")

    # 4) 保存 4 项输出
    base = Path(out_dir) / prefix
    path_primary = str(base) + "_primary.png"
    path_refined = str(base) + "_refined.png"
    path_cot = str(base) + "_cot.txt"
    path_re_edit = str(base) + "_re_edit.txt"
    primary_image.save(path_primary)
    refined_image.save(path_refined)
    with open(path_cot, "w", encoding="utf-8") as f:
        f.write(cot)
    with open(path_re_edit, "w", encoding="utf-8") as f:
        f.write(re_edit_instruction)

    logger.info(f"Saved: {path_primary}, {path_refined}, {path_cot}, {path_re_edit}")
    return {
        "primary_image": primary_image,
        "refined_image": refined_image,
        "cot": cot,
        "re_edit_instruction": re_edit_instruction,
        "paths": {
            "primary": path_primary,
            "refined": path_refined,
            "cot": path_cot,
            "re_edit": path_re_edit,
        },
    }
