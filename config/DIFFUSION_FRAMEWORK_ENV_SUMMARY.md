# Diffusion Framework Environment Requirements

Environment dependencies, execution modes, and configuration requirements for all diffusion frameworks supported in `config_iterative_refinement.yaml`.

## Table of Contents

- [Overview](#overview)
- [Framework Details](#framework-details)
- [Classification by Execution Mode](#classification-by-execution-mode)
- [External Repository Requirements](#external-repository-requirements)
- [Quick Reference](#quick-reference)

---

## Overview

This document covers **11 diffusion-based image editing models** supported in the RE-Edit pipeline, categorized by:
- **Execution Mode**: Multiprocessing (same env) vs. Subprocess (isolated conda env)
- **External Dependencies**: Built-in libraries vs. external Git repositories
- **GPU Support**: Multi-GPU parallelism capabilities

---

## Framework Details

### 1. DreamOmni2

| Property | Value |
|----------|-------|
| **Type** | `dreamomni2` |
| **Execution Mode** | **Subprocess only** (requires `conda_env` parameter) |
| **Environment** | Can reuse main environment (set `conda_env: "RE-Edit"`) |
| **Model Components** | 3 model paths required:<br>• VLM model<br>• Base Diffusion model<br>• Edit LoRA weights |
| **External Repository** | **Required**: `dreamomni2_repo` |
| **GPU** | Multi-GPU supported |

**Configuration Example**:

```yaml
vlm_path: "/path/to/DreamOmni2/vlm-model"
base_model_path: "/path/to/FLUX.1-Kontext-dev"
edit_lora_path: "/path/to/DreamOmni2/edit_lora"
dreamomni2_repo: "/path/to/DreamOmni2"  # Git repo
conda_env: "RE-Edit"  # Reuse main environment
```

**Setup**:

```bash
git clone https://github.com/YourOrg/DreamOmni2.git
```

---

### 2. FLUX.1-Kontext

| Property | Value |
|----------|-------|
| **Type** | `flux_kontext` |
| **Execution Mode** | Multi-threading (same process) |
| **Environment** | Same as main process |
| **Model Source** | HuggingFace `black-forest-labs/FLUX.1-Kontext-dev` or local path |
| **GPU** | Multi-GPU via multi-threading |

---

### 3. FLUX.2-dev

| Property | Value |
|----------|-------|
| **Type** | `flux2_dev` |
| **Execution Mode** | Multiprocessing |
| **Environment** | Same as main process |
| **Model Source** | HuggingFace `black-forest-labs/FLUX.2-dev` or local path |
| **GPU** | Multi-GPU supported; `use_cpu_offload` available for memory saving |

---

### 4. HiDream-E1.1

| Property | Value |
|----------|-------|
| **Type** | `hidream_e1` |
| **Execution Mode** | **Subprocess only** (requires `conda_env` parameter) |
| **Environment** | Can reuse main environment (set `conda_env: "RE-Edit"`) |
| **Model Components** | 3 model paths required:<br>• LLaMA base model<br>• HiDream-I1 base model<br>• HiDream-E1 editing weights |
| **External Repository** | **Required**: `hidream_repo` (contains `pipeline_hidream_image_editing.py`) |
| **GPU** | Multi-GPU supported |

**Configuration Example**:

```yaml
llama_path: "/path/to/Meta-Llama-3.1-8B-Instruct"
hidream_i1_path: "/path/to/HiDream-I1-Full"
hidream_e1_path: "/path/to/HiDream-E1-1"
hidream_repo: "/path/to/HiDream-E1"  # Git repo
conda_env: "RE-Edit"  # Reuse main environment
```

---

### 5. Janus-4o-7B

| Property | Value |
|----------|-------|
| **Type** | `janus` |
| **Execution Mode** | **Subprocess only** (requires `conda_env` parameter) |
| **Environment** | Can reuse main environment (set `conda_env: "RE-Edit"`) |
| **Core Dependencies** | `janus.models` (VLChatProcessor, MultiModalityCausalLM), `transformers`, `torch` |
| **External Repository** | **Required**: `janus_repo` (contains `janus/models`) |
| **Model Source** | HuggingFace `FreedomIntelligence/Janus-4o-7B` or local path |
| **GPU** | Multi-GPU supported |

**Setup**:

```bash
git clone https://github.com/deepseek-ai/Janus.git
# Set janus_repo: "/path/to/Janus" and conda_env: "RE-Edit" in config
```

---

### 6. OmniGen2

| Property | Value |
|----------|-------|
| **Type** | `omnigen2` |
| **Execution Mode** | **Subprocess only** (requires `conda_env` parameter) |
| **Environment** | Can reuse main environment (set `conda_env: "RE-Edit"`) |
| **External Repository** | **Required**: `omnigen2_repo` |
| **Model Source** | HuggingFace `OmniGen2/OmniGen2` or local path |
| **GPU** | Multi-GPU supported |

**Setup**:

```bash
git clone https://github.com/VectorSpaceLab/OmniGen2.git
# Set omnigen2_repo: "/path/to/OmniGen2" and conda_env: "RE-Edit" in config
```

---

### 7. Ovis-U1-3B

| Property | Value |
|----------|-------|
| **Type** | `ovis_u1` |
| **Execution Mode** | **Subprocess only** (`conda_env` parameter required) |
| **Environment** | **Dedicated conda environment required** (follow official setup) |
| **External Repository** | Optional `ovis_repo` (for custom code) |
| **Model Source** | HuggingFace `AIDC-AI/Ovis-U1-3B` or local path |
| **GPU** | Multi-GPU supported |

**Setup** (Official Instructions):

```bash
git clone https://github.com/AIDC-AI/Ovis-U1.git
conda create -n ovis-u1 python=3.10 -y
conda activate ovis-u1
cd Ovis-U1
pip install -r requirements.txt
pip install -e .
pip install "flash-attn<2.5" --no-build-isolation
```

---

### 8. Qwen-Image-Edit

| Property | Value |
|----------|-------|
| **Type** | `multi_gpu_qwen_edit` |
| **Execution Mode** | Multiprocessing |
| **Environment** | Same as main process |
| **Model Source** | HuggingFace or local path |
| **GPU** | Multi-GPU supported (one process per GPU) |

---

### 9. Qwen-Image-Edit-2511

| Property | Value |
|----------|-------|
| **Type** | `qwen_image_edit_2511` |
| **Execution Mode** | Multiprocessing or Subprocess |
| **Environment** | Same as main process (or separate conda env if using subprocess) |
| **Model Source** | HuggingFace or local path |
| **GPU** | Multi-GPU supported |

---

### 10. Step1X-Edit v1p1

| Property | Value |
|----------|-------|
| **Type** | `step1x_edit_v1p1` |
| **Execution Mode** | Subprocess (`conda_env` parameter required) |
| **Environment** | **Dedicated conda environment required** (follow official setup) |
| **Model Source** | `stepfun-ai/Step1X-Edit-v1p1-diffusers` or local path |
| **GPU** | Multi-GPU supported |

**Setup** (Official Instructions):

```bash
git clone -b dev/MergeV1-2 https://github.com/Peyton-Chen/diffusers.git
cd diffusers
pip install -e .
pip install transformers==4.55.0
```

---

### 11. Step1X-Edit v1p2 Preview

| Property | Value |
|----------|-------|
| **Type** | `step1x_edit_v1p2_preview` |
| **Execution Mode** | Subprocess (`conda_env` parameter required) |
| **Environment** | **Dedicated conda environment required** (follow official setup) |
| **Model Source** | `stepfun-ai/Step1X-Edit-v1p2-preview` or local path |
| **GPU** | Multi-GPU supported |

**Setup** (Official Instructions):

```bash
git clone -b step1xedit https://github.com/Peyton-Chen/diffusers.git
cd diffusers
pip install -e .
pip install transformers==4.55.0
```

---

## Classification by Execution Mode

### Multiprocessing / Same Process
**No separate conda environment required** - runs in main process environment:

- `multi_gpu_qwen_edit` - Qwen-Image-Edit
- `qwen_image_edit_2511` - Qwen-Image-Edit-2511
- `flux_kontext` - FLUX.1-Kontext
- `flux2_dev` - FLUX.2-dev

### Subprocess Only
**Requires `conda_env` parameter** - runs in isolated process:

| Framework | Dedicated Env Required | Can Reuse Main Env |
|-----------|------------------------|---------------------|
| `step1x_edit_v1p1` | ✅ Yes | ❌ No |
| `step1x_edit_v1p2_preview` | ✅ Yes | ❌ No |
| `ovis_u1` | ✅ Yes | ❌ No |
| `janus` | ❌ No | ✅ Yes* |
| `hidream_e1` | ❌ No | ✅ Yes* |
| `omnigen2` | ❌ No | ✅ Yes* |
| `dreamomni2` | ❌ No | ✅ Yes* |

**\*Note**: For models that can reuse the main environment, set `conda_env: "RE-Edit"` (or your main environment name) in the config. The subprocess mechanism is still required for these models, but no separate conda environment setup is needed.

---

## Quick Reference

| # | Framework | Type | Execution | Conda Env | External Repo |
|---|-----------|------|-----------|-----------|---------------|
| 1 | Qwen-Image-Edit | `multi_gpu_qwen_edit` | Multiprocessing | Same | ❌ |
| 2 | Qwen-Image-Edit-2511 | `qwen_image_edit_2511` | Multi/Subprocess | Same/Dedicated | ❌ |
| 3 | Step1X-Edit v1p1 | `step1x_edit_v1p1` | Subprocess | **Dedicated** | ❌ |
| 4 | Step1X-Edit v1p2 | `step1x_edit_v1p2_preview` | Subprocess | **Dedicated** | ❌ |
| 5 | FLUX.1-Kontext | `flux_kontext` | Multi-threading | Same | ❌ |
| 6 | FLUX.2-dev | `flux2_dev` | Multiprocessing | Same | ❌ |
| 7 | Janus-4o-7B | `janus` | Subprocess | Same | ✅ |
| 8 | Ovis-U1-3B | `ovis_u1` | Subprocess | **Dedicated** | Optional |
| 9 | HiDream-E1.1 | `hidream_e1` | Subprocess | Same | ✅ |
| 10 | OmniGen2 | `omnigen2` | Subprocess | Same | ✅ |
| 11 | DreamOmni2 | `dreamomni2` | Subprocess | Same | ✅ |

**Legend**:
- **Execution**: Multiprocessing (same process env), Multi-threading (same process), Subprocess (isolated process)
- **Conda Env**: 
  - **Same** (reuse main env): Set `conda_env: "RE-Edit"` in config
  - **Dedicated** (requires separate conda env setup): Set `conda_env: "<your_dedicated_env_name>"` after following official setup
- **External Repo**: ✅ Required, ❌ Not required, Optional

---

## Notes

1. **Subprocess models** always require the `conda_env` parameter in config, even if reusing the main environment.
   - **For models that can reuse main env** (Janus, HiDream-E1, OmniGen2, DreamOmni2): Set `conda_env: "RE-Edit"` (or your main environment name).
   - **For models requiring dedicated env** (Step1X-Edit v1p1/v1p2, Ovis-U1): Set `conda_env` to the dedicated environment name after following official setup instructions.
2. **External repositories** must be cloned and their paths specified in the model's `params` section.
3. All models support **multi-GPU parallelism** via `device_ids` parameter.
