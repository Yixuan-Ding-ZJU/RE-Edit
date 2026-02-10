#!/usr/bin/env python
import argparse
import logging
import sys
from pathlib import Path

# 项目根加入 path，供 editrefine_inference 和 src 使用
PROJECT_ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from editrefine_inference.config_loader import load_editrefine_config
from editrefine_inference.runner import run_single


def main():
    parser = argparse.ArgumentParser(
        description="EditRefine single-sample inference: image + instruction → primary edit → MLLM → one-step refinement → save 4 outputs"
    )
    parser.add_argument("--editrefine-config", type=str, required=True,
                        help="Path to EditRefine config YAML (e.g. config_editrefine_inference.yaml)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--instruction", type=str, required=True,
                        help="Editing instruction text")
    parser.add_argument("--original-description", type=str, default="",
                        help="Optional original image description for MLLM")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (overrides config editrefine.output_dir)")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Output file name prefix (default from config or 'editrefine')")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    config = load_editrefine_config(args.editrefine_config)

    result = run_single(
        config=config,
        image=image,
        edit_instruction=args.instruction,
        original_description=args.original_description,
        output_dir=args.output_dir,
        output_name_prefix=args.output_prefix,
    )

    print("EditRefine inference done. Outputs:")
    for k, v in result["paths"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
