# PhysTrace: Interpretable Temporal Evaluation of Physical Consistency in Video Generative Models

<p align="center">
  <a href="https://arxiv.org/abs/2610.31429">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="arXiv">
  </a>
</p>

Official codebase for the paper **"[PhysTrace: Interpretable Temporal Evaluation of Physical Consistency in Video Generative Models](https://arxiv.org/abs/2610.31429)"**.

<div>
<!-- TODO: replace with the teaser figure -->
<img width="100%" alt="PhysTrace Overview" src="https://github.com/user-attachments/assets/b5a56a9b-83f4-45c2-b4e5-917a2ba8d0e7" />
</div>

## Overview

This repository provides the code for **PhysTrace**, a training-free framework that assesses physical consistency by tracking the alignment between history-conditioned expectations and video observations over time.

- **Expectation tracking**: for each window, an MLLM (Qwen3.8-27B by default) formulates an Expected State from past observations, which is compared with the video in a shared embedding space (PE-AV-Large by default) to produce a time-resolved alignment trace.
- **IntPhys2-Timing**: temporal violation annotations (onset / end) on top of IntPhys2 for additionally evaluating temporal localization.
- **Reference-guided masked denoising**: generates videos from mixed possible–impossible latents to assess how pretrained video generators resolve the two outcomes.

## 1. Repository Structure

```text
project_root/
├── AlignmentModels/      # video-text alignment / MLLM code (submodules) and weights
├── DiffusionModels/      # video generative model code (submodules) and weights
├── IntPhys2Timing/       # violation GT CSVs and per-attribute prompts
├── PhysTrace/            # core library (history-conditioned expectation and alignment scoring)
├── demo/                 # interactive web demo
├── score/                # batch inference and scoring
├── requirements.txt
└── README.md
```

### Main entry points

- `demo/app.py`: interactive demo (browser UI) for inspecting per-window results
- `score/run_test.py`: batch inference over every IntPhys2 pair
- `score/run_eval.py`: scoring of the inference results against the IntPhys2-Timing GT

## 2. Installation

### a) Environment

Tested on **NVIDIA A100** with `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`.

```bash
pip3 install -r requirements.txt
```

### b) Code Download

```bash
git clone --recursive <REPO_URL>
# or, after a plain clone
git submodule update --init --remote --recursive
```

Or clone the external repositories manually:

```bash
# Alignment Scoring Models
git clone https://github.com/facebookresearch/perception_models.git AlignmentModels/PE_AV_large

# Diffusion/flow-based video generative models (only for Section 4: Evaluation for Video Generative Models)
git clone https://github.com/Wan-Video/Wan2.1.git DiffusionModels/Wan2.1
git clone https://github.com/nvidia-cosmos/cosmos-predict2.git DiffusionModels/cosmos-predict2

# (Optional) Diffusion/flow-based video generative models
git clone https://github.com/meituan-longcat/LongCat-Video.git DiffusionModels/LongCat-Video
git clone https://github.com/AILab-CVC/VideoCrafter.git DiffusionModels/VideoCrafter
```

### c) Weight Download

```bash
hf auth login
```

**Alignment Scoring Models**

```bash
hf download facebook/pe-av-large --revision perception_models --local-dir AlignmentModels/weights/PE-AV-Large
```

**MLLMs**

```bash
hf download Qwen/Qwen3.8-27B-FP8 --local-dir AlignmentModels/weights/Qwen3.8-27B-fp8
```

**Diffusion/flow-based video generative models** (only for [4. Evaluation for Video Generative Models](#4-evaluation-for-video-generative-models))

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir DiffusionModels/weights/Wan2.1-T2V-1.3B

hf download nvidia/Cosmos-Predict2-2B-Video2World --include "model-480p-16fps.pt" "config.json" "model_index.json" "scheduler/*" "text_encoder/*" "tokenizer/*" "vae/*" --local-dir DiffusionModels/weights/Cosmos-Predict2-2B-Video2World
```

**Optional Models**

```bash
# ---- Scorer : internvideo2-stage2-6b ----
# internvideo2-stage2
hf download OpenGVLab/InternVideo2-Stage2_6B-224p-f4 internvideo2-s2_6b-224p-f4_with_audio_encoder.pt \
    --local-dir AlignmentModels/weights/InternVideo2-Stage2

# bert tokenizer
hf download google-bert/bert-large-uncased vocab.txt tokenizer.json tokenizer_config.json config.json \
    --local-dir AlignmentModels/weights/InternVideo2-Stage2/bert-large-uncased

# ---- Scorer : internvideo2-clip-6b ----
# internvideo2-stage2 (same as above; skip if already downloaded)
hf download OpenGVLab/InternVideo2-Stage2_6B-224p-f4 internvideo2-s2_6b-224p-f4_with_audio_encoder.pt \
    --local-dir AlignmentModels/weights/InternVideo2-Stage2

# InternVL-C 13B (25GB)
hf download OpenGVLab/InternVL internvl_c_13b_224px.pth \
    --local-dir AlignmentModels/weights/InternVideo2-CLIP

# CLIP delta
hf download OpenGVLab/InternVideo2-CLIP-6B-224p-f8 6B_clip.pth \
    --local-dir AlignmentModels/weights/InternVideo2-CLIP

# LLaMA tokenizer
hf download hfl/chinese-alpaca-lora-7b tokenizer.model tokenizer_config.json special_tokens_map.json \
    --local-dir AlignmentModels/weights/InternVideo2-CLIP/tokenizer

# ---- MLLM : Qwen3-VL-8B-Thinking ----
hf download Qwen/Qwen3-VL-8B-Thinking --local-dir AlignmentModels/weights/Qwen3-VL-8B-Thinking

# ---- MLLM : Qwen3-VL-32B-Thinking ----
hf download Qwen/Qwen3-VL-32B-Thinking --local-dir AlignmentModels/weights/Qwen3-VL-32B-Thinking

# ---- MLLM : Cosmos-Reason2-8B ----
hf download nvidia/Cosmos-Reason2-8B --local-dir AlignmentModels/weights/Cosmos-Reason2-8B

# ---- Diffusion/flow-based model : CogVideoX-5B ----
hf download zai-org/CogVideoX-5b --local-dir DiffusionModels/weights/CogVideoX-5b

# ---- Diffusion/flow-based model : LongCat-Video-13.6B ----
hf download meituan-longcat/LongCat-Video --exclude "lora/*" --local-dir DiffusionModels/weights/LongCat-Video

# ---- Diffusion/flow-based model : ModelScope-T2V-1.7B ----
hf download ali-vilab/text-to-video-ms-1.7b --exclude "*.bin" --local-dir DiffusionModels/weights/text-to-video-ms-1.7b

# ---- Diffusion/flow-based model : VideoCrafter2 ----
hf download VideoCrafter/VideoCrafter2 model.ckpt --local-dir DiffusionModels/weights/VideoCrafter2
```

### d) Data Download: IntPhys2

```bash
python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='facebook/IntPhys2', repo_type='dataset', local_dir='/Path/To/IntPhys2')"
```

## 3. Evaluation with IntPhys2-Timing

### a) Demo

```bash
CUDA_VISIBLE_DEVICES=0 python3 demo/app.py \
    --intphys2-dir /Path/To/IntPhys2 \
    --result-dir results/demo \
    --port 1535
```

- `--intphys2-dir`: IntPhys2 root (uses `/Path/To/IntPhys2/Main/Videos` and `/Path/To/IntPhys2/Main/metadata.csv`; the CLIP cache `/Path/To/IntPhys2/Main/videos-npz` is created beside them)
- `--result-dir`: where `results.csv` / `trials.csv` are written
- `--gt-csv`: violation GT drawn on the UI chart (defaults to `IntPhys2Timing/intphys2_gt.csv`)
- `--mllm-backend`: `qwen3.8-27b` (default), `qwen3-vl-thinking`, `qwen3-vl-32b-thinking`, `cosmos-reason2`
- `--clip-model`: `pe-av-large` (default), `internvideo2-stage2-6b`, `internvideo2-clip-6b`

### b) Inference

```bash
CUDA_VISIBLE_DEVICES=0 python3 score/run_test.py \
    --intphys2-dir /Path/To/IntPhys2 \
    --result-dir results/test \
    --window-seconds 2 --hop-seconds 1 \
    --clip-frames 60 --mllm-frames 60 \
    --categories solidity=BoxSoldity
```

- `--intphys2-dir`, `--result-dir`: same as the demo (same result format, so the result directory can be opened in the demo)
- `--window-seconds`, `--hop-seconds`: window length and hop between windows, in seconds
- `--clip-frames`, `--mllm-frames`: frames fed to CLIP / the MLLM per window
- `--categories`: limit the categories run per attribute, e.g. `solidity=BoxSoldity` (default: every category)
- `--mllm-backend`: `qwen3.8-27b` (default), `qwen3-vl-thinking`, `qwen3-vl-32b-thinking`, `cosmos-reason2`
- `--clip-model`: `pe-av-large` (default), `internvideo2-stage2-6b`, `internvideo2-clip-6b`

### c) Scoring

```bash
python3 score/run_eval.py --result-dir results/test
```

- `--result-dir`: result directory of the inference above
- `--strict-edges`: a band that only touches a GT edge is not counted as a hit (optional)
- `--out`: output file stem (default: `<result-dir>/eval_gt`)

> **Note**  
> `score/run_test.py` is resumable. Re-running the same command skips pairs that are already stored; use `--status` to print progress and `--force` to re-analyze.

## 4. Evaluation for Video Generative Models

For each pair of the full IntPhys2-Timing dataset, a video generative model regenerates the masked frames from the violation onset onward, guided toward the possible or impossible clip, and the results are evaluated with PhysTrace. Run each model with its own output directory.

### a) Video Generation

**Wan2.1-T2V-1.3B**

```bash
CUDA_VISIBLE_DEVICES=0 python3 PhysTrace/gen_wan.py \
    -i /Path/To/IntPhys2 \
    -o results/wan \
    --gt-csv IntPhys2Timing/intphys2_gt_full.csv \
    --sigmas 0.8 --lambdas 0.5 --seeds 0,1 --prefix I --sigma-hard 1 \
    --overlap half
```

**Cosmos-Predict2-2B**

```bash
CUDA_VISIBLE_DEVICES=0 python3 PhysTrace/gen_cosmos20.py \
    -i /Path/To/IntPhys2 \
    -o results/cosmos \
    --gt-csv IntPhys2Timing/intphys2_gt_full.csv \
    --sigmas 0.8 --lambdas 0.5 --seeds 0,1 --prefix I --sigma-hard 1 \
    --overlap half
```

- `-i`: IntPhys2 root (same as `--intphys2-dir` above)
- `-o`: output directory; each pair is written to `<cid>/resp_l<lambda>_s<sigma>_p<prefix>_seed<n>.mp4` with a `meta.json`, and each output setting is also collected into `sets/<output name>/Main` in the IntPhys2 layout
- `--gt-csv`: GT CSV of the pairs to generate (`intphys2_gt_full.csv` = the full IntPhys2-Timing dataset)
- `--sigmas`: comma-separated start levels of the sampling (flow sigma; `1.0` = pure noise)
- `--lambdas`: comma-separated guidance strengths (`1.0` = overwrite the x0 estimate with the chosen side)
- `--seeds`: comma-separated seeds
- `--prefix`: clip providing the frames before the violation onset, `I` (Impossible) or `P` (Possible)
- `--sigma-hard`: fraction of the steps after which the choice between P and I is fixed (`1` = only the last step)
- `--overlap`: frames shared by consecutive windows (`0` = none; `half` = 41 of 81 frames for Wan, 45 of 93 frames for Cosmos)

### b) Physically Impossible Classification Rates

Run [3. b) Inference](#b-inference) twice with the same settings, once on a generated set and once on the original videos as the reference:

- Generated: `--intphys2-dir results/wan/sets/resp_l0.5_s0.80_pI_seed0 --result-dir results/test_wan_l0.5_s0.80_seed0`
- Reference: `--intphys2-dir /Path/To/IntPhys2 --result-dir results/test_orig`

Then compare the two runs:

```bash
python3 score/run_compare.py \
    --result-dir results/test_wan_l0.5_s0.80_seed0 \
    --reference-dir results/test_orig \
    --gt-csv IntPhys2Timing/intphys2_gt_full.csv
```

- `--result-dir`: inference result of the generated set
- `--reference-dir`: inference result of the original videos
- `--gt-csv`: GT CSV of the full IntPhys2-Timing dataset, the same one used for the generation

A generated clip is classified as impossible when its lowest window score is below that of the pair's original Possible clip. The report is written to `<result-dir>/eval_compare.json`.

## Citation

If you find this repository useful, please cite the paper.

```bibtex
@article{yoo2026phystrace,
    title={{PhysTrace}: Interpretable Temporal Evaluation of Physical Consistency in Video Generative Models},
    author={Yoo, Taekoan and Jung, Wonkyung and Kim, Kyunghun and Kong, Kyeongbo},
    journal={arXiv preprint arXiv:2610.31429},
    year={2026},
    url={https://arxiv.org/abs/2610.31429}
}
```
