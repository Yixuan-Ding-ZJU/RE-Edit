"""
EditRefine 推理配置加载
从 base_config 加载 diffusion_model、mllm，合并 editrefine 段
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_editrefine_config(editrefine_config_path: str) -> Dict[str, Any]:
    """
    加载 EditRefine 推理用完整配置。
    先读 editrefine 配置文件，再根据 base_config 路径加载基础配置并合并。

    Args:
        editrefine_config_path: config_editrefine_inference.yaml 路径

    Returns:
        合并后的配置字典，包含 diffusion_model、mllm、editrefine 等
    """
    path = Path(editrefine_config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"EditRefine config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        editrefine_file = yaml.safe_load(f)

    base_config_key = editrefine_file.get("base_config", "config_iterative_refinement.yaml")
    base_path = path.parent / base_config_key
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # 合并：以 base 为主，editrefine 段覆盖/追加
    merged = dict(base_config)
    if "editrefine" in editrefine_file:
        merged["editrefine"] = editrefine_file["editrefine"]
    else:
        merged["editrefine"] = {}

    return merged
