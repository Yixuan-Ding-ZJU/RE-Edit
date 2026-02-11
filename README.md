# WiseEdit: Benchmarking Cognition- and Creativity-Informed Image Editing

**Kaihang Pan\*** · **Weile Chen\*** · Haiyi Qiu\* · Qifan Yu · Wendong Bu · Zehan Wang ·  
Yun Zhu · Juncheng Li · Siliang Tang  

<sup>1</sup>Zhejiang University  
<sup>2</sup>Shanghai Artificial Intelligence Laboratory  

\* Equal contribution.

---

## 🔗 Links

[![arXiv](https://img.shields.io/badge/arXiv-2512.00387-b31b1b.svg)](https://arxiv.org/abs/2512.00387)
[![Homepage](https://img.shields.io/badge/Homepage-Project-green)](https://your-project-page.com)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/yourname)
[![Dataset](https://img.shields.io/badge/Dataset-WiseEdit-orange)](https://huggingface.co/datasets/yourname/WiseEdit)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/yourname/wiseedit)

---


# RE-Edit & EditRefine 

Image editing model evaluation system for testing reasoning-aware capabilities on RE-Edit benchmark and EditRefine module effectiveness.

## Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [RE-Edit Pipeline](#re-edit-pipeline)
- [EditRefine Standalone Inference](#editrefine-standalone-inference)
- [Configuration Reference](#configuration-reference)
- [Extension Guide](#extension-guide)

---

## Project Structure

```
RE-Edit_EditRefine/
├── README.md                                 # Documentation
├── requirements.txt                          # Dependencies
├── main.py                                   # RE-Edit Pipeline entry
├── run_editrefine_inference.py               # EditRefine Inference entry
│
├── config/                                   # EditRefine standalone module
│   ├── config_iterative_refinement.yaml      # RE-Edit Pipeline config
│   ├── config_editrefine_inference.yaml      # EditRefine Inference config
│   └── DIFFUSION_FRAMEWORK_ENV_SUMMARY.md
│
├── editrefine_inference/                     # EditRefine standalone module
│   ├── __init__.py
│   ├── config_loader.py
│   └── runner.py
│
└── src/                                      # Source code
    ├── pipeline.py
    ├── iterative_pipeline_v7.py              # Pipeline implementation
    ├── data/                                 # Data loading
    │   ├── benchmark_loader.py
    │   ├── iterative_data.py
    │   └── data_types.py
    ├── models/                               # Models
    │   ├── diffusion/                        # Image editing models (11 types)
    │   │   ├── base_diffusion.py
    │   │   └── implementations/
    │   ├── mllm/                             # MLLM for analysis cot & re-edit
    │   │   ├── base_mllm.py
    │   │   └── implementations/
    │   └── reward/                           # Reward models
    │       ├── base_reward.py
    │       └── implementations/
    ├── evaluation/                           # Evaluation & reporting
    │   ├── scorer.py
    │   └── reporter.py
    └── utils/                                # Utilities
        ├── image_utils.py
        ├── logger.py
        └── prompt_manager.py
```

---

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/Yixuan-Ding-ZJU/RE-Edit.git
conda create -n RE-Edit python==3.12
conda activate RE-Edit
cd RE-Edit
pip install -r requirements.txt
```

### 2. RE-Edit Pipeline (Full Evaluation)

```bash
# Edit config to select model & settings
nano config/config_iterative_refinement.yaml

# Run evaluation
python main.py --config config/config_iterative_refinement.yaml --mode iterative
```

### 3. EditRefine Standalone Inference (Single Image)

```bash
python run_editrefine_inference.py \
  --editrefine-config config/config_editrefine_inference.yaml \
  --image /path/to/image.png \
  --instruction "Add a red hat"
```

---

## RE-Edit Pipeline

Full evaluation pipeline for RE-Edit benchmark with 5 stages.

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| **Stage 1** | Primary Editing: initial edit with target diffusion model |
| **Stage 2** | EditRefine Reasoning Agent: analyze result, generate CoT reasoning & re-edit instruction |
| **Stage 3** | EditRefine Executor Engine: refine with re-edit instruction |
| **Stage 4** | Comparative Scoring: evaluate both primary & refined images |
| **Stage 5** | Statistics: aggregate metrics & generate report |

### Key Configuration

**Evaluation Settings**:

```yaml
evaluation:
  output_dir: "./results_iterative"
  save_images: true
  primary_images_dir: null              # Skip Stage 1 if non-empty , load primary image from dir 
  primary_image_suffix: "_primary.png"
  skip_stage4: false                     # Skip scoring if true
  ############################# Key Point #############################
  skip_refinement: false                 # Skip EditRefine (Stage 2-3) if true, just perform evaluation of specific image edit model on RE-Edit
```

**Diffusion Models** (11 types supported, detailed see [`config/DIFFUSION_FRAMEWORK_ENV_SUMMARY.md`](config/DIFFUSION_FRAMEWORK_ENV_SUMMARY.md)):

```yaml
diffusion_model:
  primary:                               # Model under evaluation
    type: step1x_edit_v1p1               # Options: multi_gpu_qwen_edit, flux2_dev,
                                         #   step1x_edit_v1p1, step1x_edit_v1p2_preview,
                                         #   janus, ovis_u1, hidream_e1, omnigen2,
                                         #   flux_kontext, dreamomni2, qwen_image_edit_2511
    params:
      model_name: "/path/to/model"
      device_ids: [0, 1, 2, 3]
      seed: 42
      num_inference_steps: 28

  refinement:                            # Fixed two EditRefine Executor Engines
    type: multi_gpu_qwen_edit
    params:
      model_name: "/path/to/qwen-edit"
      device_ids: [0, 1, 2, 3]
      seed: 42
      num_inference_steps: 1
```

**MLLM** (Reasoning Agent):

```yaml
mllm:
  type: qwen25_vl
  params:
    model_name: "/path/to/qwen2.5-vl"
    device: "auto"
    batch_size: 16
    max_new_tokens: 512
```

**Reward Model** (vLLM recommended for speed):

```yaml
reward_model:
  type: qwen3_vl_vllm_subprocess
  params:
    model_name: "/path/to/Qwen3-VL-30B"
    tensor_parallel_size: 4               # Must be divisor of 32 (attn heads)
    batch_size: 8
    conda_env: "yx_vllm"
    timeout: 1200
```

---

## EditRefine Standalone Inference

Single-image inference: **Image + Instruction** → Primary Edit → EditRefine Reasoning Agent Analysis → EditRefine Execution Engine One-step Refinement → Save 4 outputs.

### Features

- **Config**: `config_editrefine_inference.yaml` references `base_config: config_iterative_refinement.yaml` (reuses `diffusion_model`, `mllm`)
- **Outputs**: 4 files per run
  - `{prefix}_primary.png` - primary edited image
  - `{prefix}_refined.png` - refined edited image by EditRefine
  - `{prefix}_cot.txt` - chain-of-thought reasoning
  - `{prefix}_re_edit.txt` - re-edit instruction
- **Module**: `editrefine_inference/` (`config_loader`, `runner`)

### Usage


**With Custom Output**:

```bash
python run_editrefine_inference.py \
  --editrefine-config config/config_editrefine_inference.yaml \
  --image img.png \
  --instruction "Change the sky to sunset" \
  --output-dir ./my_output \
  --output-prefix experiment_01
```

**Optional Arguments**:
- `--output-dir` - override `editrefine.output_dir` in config
- `--output-prefix` - output filename prefix (default: "editrefine")

### How to Switch Image Edit Model

Edit `config/config_iterative_refinement.yaml` and uncomment desired model in `diffusion_model.primary` section. 11 models supported (see [`config/DIFFUSION_FRAMEWORK_ENV_SUMMARY.md`](config/DIFFUSION_FRAMEWORK_ENV_SUMMARY.md) for environment requirements).

---

## Configuration Reference

### Diffusion Models

**11 models supported**:
- `multi_gpu_qwen_edit` - Qwen-Image-Edit 
- `qwen_image_edit_2511` - Qwen-Image-Edit-2511
- `step1x_edit_v1p1` - Step1X-Edit v1p1 
- `step1x_edit_v1p2_preview` - Step1X-Edit v1p2 
- `flux_kontext` - FLUX.1-Kontext
- `flux2_dev` - FLUX.2-dev
- `janus` - Janus-4o-7B 
- `ovis_u1` - Ovis-U1-3B 
- `hidream_e1` - HiDream-E1.1 
- `omnigen2` - OmniGen2 
- `dreamomni2` - DreamOmni2 


### Evaluation Metrics

Control which metrics to evaluate:

```yaml
evaluation:
  enable_sc_metric: true                 # Semantic Consistency
  enable_instruction_following_metric: true  # Instruction Following
  enable_primary_scoring: true          # Score primary images (compute improvement_rate)

```

---

## Extension Guide

### Add New Diffusion Model

1. Create implementation in `src/models/diffusion/implementations/`
2. Inherit from `BaseDiffusionModel`
3. Implement `edit_image()` and optionally `batch_edit()`
4. Register in `iterative_pipeline_v7.py` loaders
5. Add config template to `config/config_iterative_refinement.yaml`

### Add New Reward Model

1. Create implementation in `src/models/reward/implementations/`
2. Inherit from `BaseRewardModel`
3. Implement `score()` method
4. Register in pipeline loader

---

## License

MIT License
