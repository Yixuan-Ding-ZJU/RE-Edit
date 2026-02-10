#!/bin/bash
# 使用示例脚本

# 设置路径
JSON_PATH="/mnt/autodl_tmp1/dyx/Benchmark/version_4_simplified_v2_translated_with_images.json"
IMAGE_DIR="/mnt/autodl_tmp1/dyx/image_edit_benchmark_12_14_grab_from_h100/results_iterative/ICML_12_18_flux_2"
CONFIG_PATH="/mnt/autodl_tmp1/dyx/image_edit_benchmark_12_14_grab_from_h100/single_scoring_part_vllm/scoring_config.yaml"
CHECKPOINT_PATH="/mnt/autodl_tmp1/model/models/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/4b184fbdab8886057d8d80c09f35bcfc65fe640e"

# 示例1: 评分物理维度
echo "示例1: 评分物理维度"
python main.py \
    --json_path "$JSON_PATH" \
    --image_dir "$IMAGE_DIR" \
    --config_path "$CONFIG_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --metric_type 物理 \
    --batch_size 3 \
    --max_new_tokens 128

# 示例2: 评分PQ指标
# echo "示例2: 评分PQ指标"
# python main.py \
#     --json_path "$JSON_PATH" \
#     --image_dir "$IMAGE_DIR" \
#     --config_path "$CONFIG_PATH" \
#     --checkpoint_path "$CHECKPOINT_PATH" \
#     --metric_type pq_metric \
#     --batch_size 3 \
#     --max_new_tokens 256

# 示例3: 评分SC指标
# echo "示例3: 评分SC指标"
# python main.py \
#     --json_path "$JSON_PATH" \
#     --image_dir "$IMAGE_DIR" \
#     --config_path "$CONFIG_PATH" \
#     --checkpoint_path "$CHECKPOINT_PATH" \
#     --metric_type sc_metric \
#     --batch_size 3 \
#     --max_new_tokens 256

